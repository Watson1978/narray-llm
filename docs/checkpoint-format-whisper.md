# Whisper のチェックポイント形式

推測ではなく現物を読んで確定させたもの。出典は [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) の配布ファイル (2026-09-20 取得) と、`transformers` 5.16.1 の `models/whisper/modeling_whisper.py`。

**このモデルも参照が C ではない。** 単一ファイルの C 実装が無いので、形式の出どころは HF の配布物そのものになる (`whisper.cpp` は ggml の枠組みで、llama2.c のようには読めない)。

## 配布されているもの

| ファイル | 大きさ | |
|---|---|---|
| `model.safetensors` | 151,061,672 B | **変換が要らない。**`lib/narray_llm/safetensors.rb` がそのまま読む |
| `config.json` | 1,983 B | 下の表の出どころ |
| `preprocessor_config.json` | 184,990 B | メルフィルタが実体で入っている |
| `tokenizer.json` / `vocab.json` / `merges.txt` | 2.5 MB / 836 KB / 494 KB | GPT-2 系の BPE |

**`pytorch_model.bin` も `flax_model.msgpack` も `tf_model.h5` も配られているが、落とさない。** safetensors があるので要らない。

## 中身 (現物を読んで確認)

| | |
|---|---|
| テンソル | **167 本、全部 F32** |
| パラメータ | **37,760,640 (144.0 MiB)** |

**埋め込みが 51865 x 384 = 19,916,160 で、全体の 52.7% を占める。**

### `k_proj` にだけ bias が無い

```
self_attn.q_proj.weight   [384, 384]   self_attn.q_proj.bias   [384]
self_attn.k_proj.weight   [384, 384]   bias 無し
self_attn.v_proj.weight   [384, 384]   self_attn.v_proj.bias   [384]
self_attn.out_proj.weight [384, 384]   self_attn.out_proj.bias [384]
```

**encoder の self-attention でも、decoder の self / cross でも同じ。**`modeling_whisper.py` が `k_proj` だけ `bias=False` で作る。

### 出力射影はチェックポイントに無い

`proj_out` という名前のテンソルが無く、**`model.decoder.embed_tokens.weight` を逆向きに読む**。Switch が 4 つの名前で 1 つの記憶域を指していたのと同じ話だが、**こちらは名前そのものが無い**。

### 位置は 2 つとも実体で入っている

```
model.encoder.embed_positions.weight   [1500, 384]   正弦波 (論文では生成するもの)
model.decoder.embed_positions.weight   [448, 384]    学習済み
```

**encoder 側は生成しなくてよい。**`modeling_whisper.py` の `sinusoids()` は初期化に使うだけで、読み込み時には保存された値が入る。

## 層の構造 (`config.json`)

| | |
|---|---|
| `d_model` / heads / `ffn_dim` | 384 / 6 / 1536 (head_dim は 64) |
| encoder / decoder の層数 | 4 / 4 |
| `num_mel_bins` / `max_source_positions` | **80 / 1500** |
| `max_target_positions` / `max_length` | 448 / 448 |
| 語彙 | 51865 |
| 活性 / 正規化 | **gelu / LayerNorm** (weight と bias の両方を持つ) |
| `scale_embedding` | false |
| `decoder_start_token_id` / eos / pad / bos | 50258 / 50257 / 50257 / 50257 |
| `forced_decoder_ids` | `[[1, 50259], [2, 50359], [3, 50363]]` |
| `suppress_tokens` / `begin_suppress_tokens` | 87 個 / `[220, 50257]` |

**メルのフレーム数は `max_source_positions` の 2 倍の 3000 になる。** 2 つ目の畳み込みが stride 2 だからで、`config.json` には 3000 という数字が無い。

### 畳み込み

```
model.encoder.conv1.weight   [384, 80, 3]    Conv1d(80, 384, k=3, pad=1)
model.encoder.conv2.weight   [384, 384, 3]   Conv1d(384, 384, k=3, stride=2, pad=1)
```

**このリポジトリが一度も書いていない演算である。**

### 名前のかたち

```
model.{encoder,decoder}.layers.N.self_attn.{q,k,v,out}_proj.{weight,bias}
model.decoder.layers.N.encoder_attn.{q,k,v,out}_proj.{weight,bias}
model.{encoder,decoder}.layers.N.{self_attn,encoder_attn,final}_layer_norm.{weight,bias}
model.{encoder,decoder}.layers.N.fc{1,2}.{weight,bias}
model.{encoder,decoder}.layer_norm.{weight,bias}
model.{encoder,decoder}.embed_positions.weight
model.decoder.embed_tokens.weight
model.encoder.conv{1,2}.{weight,bias}
```

**内訳は 167 = 4 (decoder の層外) + 96 (decoder 24 種 x 4 層) + 7 (encoder の層外) + 60 (encoder 15 種 x 4 層)。**
