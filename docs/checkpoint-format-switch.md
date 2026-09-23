# Switch Transformer のチェックポイント形式

推測ではなく現物と transformers のソースを読んで確定させたもの。出典は [`google/switch-base-8`](https://huggingface.co/google/switch-base-8) の配布ファイル (2026-09-19 取得) と、`transformers` 5.16.1 の `models/switch_transformers/modeling_switch_transformers.py`。

**このモデルだけ、参照が C ではない。** llm.c / llama2.c / runq.c / mamba.c に相当する単一ファイルの C 実装が無いので、形式の出どころは HF の配布物そのものになる。

## 配布されているもの

| ファイル | 大きさ | |
|---|---|---|
| `pytorch_model.bin` | 1,238,895,063 B | **pickle。safetensors は配られていない** |
| `config.json` | 1,860 B | 下の表の出どころ |
| `spiece.model` | 791,656 B | SentencePiece |
| `tokenizer.json` | 2,422,095 B | |

## 中身 (現物を読んで確認)

| | |
|---|---|
| テンソル | **440 本** |
| dtype | **bf16 434 本 + fp32 6 本** |
| 名前のぶんのパラメータ | 693,361,920 |
| **実体のあるパラメータ** | **619,339,008** |

**`config.json` の `torch_dtype` は `bfloat16` で、これが正しい。** ファイルの大きさから fp32 だと見積もったが外れた。

### fp32 の 6 本は encoder の router だけ

```
encoder.block.{1,3,5,7,9,11}.layer.1.mlp.router.classifier.weight   [8, 768]
```

**decoder の router は bf16 である。**`config.json` の `router_dtype: float32` は **計算の型であって格納の型ではない** (`modeling_switch_transformers.py` が `hidden_states.to(self.dtype)` で毎回上げる)。

### 埋め込みは 4 つの名前が 1 つの記憶域を指す

```
shared.weight  encoder.embed_tokens.weight  decoder.embed_tokens.weight  lm_head.weight
```

**4 本とも `data_ptr()` が同じ。** 693,361,920 と 619,339,008 の差 74,022,912 が、重複 3 本ぶん (32128 x 768 x 3) である。

## 変換

**pickle を Ruby から読まない。**`python/export_switch.py` が 1 度だけ変換する。

- **fp32 に広げる。** bf16 → fp32 は情報を落とさない。**Numo に半精度が無い** ので、両バックエンドで読ませるならここで広げるしかない
- **共有された 4 本は `shared.weight` 1 本だけ書く。** 残り 3 つの名前はローダが読み替える
- 出力は **437 本、619,339,008 パラメータ、2,477,356,032 B**

`torch.equal` で 5 本を原本と突き合わせ、**すべてビット一致** を確認した。

## safetensors の読み方

`lib/narray_llm/safetensors.rb` が読む。**先頭 8 バイトがヘッダ長 (LE u64)、次が JSON ヘッダ、残りが生バイト。** 各テンソルは `dtype` / `shape` / `data_offsets [begin, end]` を持ち、行優先・リトルエンディアンで隙間も重なりも無い。

- **1 テンソルずつ seek して読む。** 2.4 GB がホストに丸ごと乗ることはない
- **隙間と重なりが無いことを読み込み前に検査する。**`data_offsets` を並べ替えて先頭から連続していること、末尾がファイル末尾と一致することを見る
- **半精度は受け付けない。**`DTYPES` に F32 / I64 / I32 しか無い。両バックエンドで読める形式に限るという判断で、Numo に bf16 が無いことがその理由

## 層の構造 (`config.json`)

| | |
|---|---|
| `d_model` / `d_ff` / `d_kv` / `num_heads` | 768 / 3072 / 64 / 12 |
| encoder / decoder の層数 | 12 / 12 |
| `encoder_sparse_step` / `decoder_sparse_step` | 2 / 2 → **sparse は奇数ブロック (1,3,5,7,9,11)**、各 6 層 |
| `num_experts` / `num_selected_experts` | 8 / 1 |
| `expert_capacity` | 64 |
| `dense_act_fn` / `is_gated_act` | relu / false |
| `layer_norm_epsilon` | 1e-6 |
| 相対位置 | bucket 32、max_distance 128。**各スタックの block 0 にしか持たない** (T5 は層で共有) |
| 語彙 / pad / eos / `decoder_start_token_id` | 32128 / 0 / 1 / 0 |

### 名前のかたち

```
{encoder,decoder}.block.N.layer.M.SelfAttention.{q,k,v,o}.weight      [768, 768]
{encoder,decoder}.block.0.layer.0.SelfAttention.relative_attention_bias.weight  [32, 12]
decoder.block.N.layer.1.EncDecAttention.{q,k,v,o}.weight             [768, 768]
{encoder,decoder}.block.N.layer.M.layer_norm.weight                  [768]
{encoder,decoder}.block.N.layer.M.mlp.{wi,wo}.weight                 dense な層
{encoder,decoder}.block.N.layer.M.mlp.router.classifier.weight       [8, 768]
{encoder,decoder}.block.N.layer.M.mlp.experts.expert_K.{wi,wo}.weight
{encoder,decoder}.final_layer_norm.weight                            [768]
shared.weight                                                         [32128, 768]
```

**`layer.M` はスタックで意味が変わる。** encoder は 0 が self-attention、1 が FF。decoder は 0 が self-attention、1 が cross-attention、2 が FF。**layer_norm が encoder は 2 本、decoder は 3 本** あるのはそのためである。
