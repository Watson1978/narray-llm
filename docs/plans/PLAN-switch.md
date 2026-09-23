# Switch Transformer 実装プラン

Switch Transformer (MoE) の推論を Numo::NArray / Cumo だけで実装する。**目的は cumo の検証** で、GPT-2 / Llama 2 / Mamba が踏んでいない経路を叩くことに価値がある。参照は Hugging Face の [`google/switch-base-8`](https://huggingface.co/google/switch-base-8) と、transformers の [`modeling_switch_transformers.py`](https://huggingface.co/docs/transformers/model_doc/switch_transformers)。

進め方は [PLAN-gpt2.md](PLAN-gpt2.md) の「全体の約束事」をそのまま継ぐ。**前の段階の受け入れテストが通るまで次の段階のコードを書かない。**

## なぜ Switch か (cumo の検証として)

| 新しく叩く経路 | これまでのモデルでは |
|---|---|
| **デバイス上で決まる添字での gather。** expert 番号は router の計算結果なので、埋め込みで使った「添字を Ruby の Array で持つ」回避策が効かない | 埋め込みの `ids` は最初からホストにある。AGENTS.md が「ONNX の Gather や MoE のエキスパート選択がそれで、別の問題として残っている」と書いている経路そのもの |
| **仕事量が入力で変わる。** expert ごとに来るトークン数が違い、**誰も来ない expert がある**。`expert_capacity` を超えたトークンは **どの expert にも行かず、残差だけで通り抜ける** | 常に全部の重みを同じだけ使う |
| **encoder-decoder。** cross-attention と、encoder 出力を跨いだ KV の持ち方 | 3 つとも decoder only |
| **T5 の相対位置バイアス** — `[num_heads, q_len, k_len]` を bucket 表から引いて scores に足す | GPT-2 は学習済み絶対位置、Llama 2 は RoPE、Mamba は位置を持たない |
| **SentencePiece (`spiece.model`)** | llm.c / llama2.c / mamba.c はどれも自前のバイナリ表を配っている |

**いちばん大きいのは 1 番目である。** そして **それは cumo 側の未解決課題でもある** — `docs/idea.md` が MoE を外した理由がここで、2026-09-19 に cumo 0.9.0 (#470 が入った版) で確かめ直しても `cumo_na_parse_narray_index` の同期警告は出たままだった。**塞がったまま着手する。** 塞がりを実際の用途で踏むこと自体が目的である。

### 中心に置く比較

**添字での gather を避ける書き方がある。** 8 個の expert を **全トークンに対して全部走らせ、マスクで選ぶ** 形で、同期は 1 回も起きない代わりに仕事量が 8 倍になる。

| | 同期 | 仕事量 | カーネル数 |
|---|---|---|---|
| (a) 振り分ける (transformers と同じ) | **expert ごとに 1 回** | 1 倍 | expert ごとに可変 |
| (b) 全部走らせてマスク | **無し** | **8 倍** | 固定 |

**どちらが速いかは分からない。** decode は 1 トークンなので `d_ff` 3072 の GEMV が 8 本走るだけで、**帯域ではなく投入で決まる帯にいる可能性がある** (Mamba で 10 点の表を作った、まさにその帯)。**これを測るのがこの実装のいちばんの成果になる見込み。**

## 関門をどうするか

**単一ファイルの C 参照が無い。** llm.c / llama2.c / runq.c / mamba.c に相当するものが Switch には無いので、**このプロジェクトで初めて、関門を作り直す** ことになる。

| | これまで | Switch |
|---|---|---|
| 参照 | C の単一ファイル | **transformers (PyTorch)** |
| 関門 | C とのトークン列完全一致 | **PyTorch とのトークン列完全一致**、加えて層ごとの中間活性の許容誤差 |

**`python/gpt2_torch.py` が既に PyTorch を参照点として使っている** ので前例はある。ただし **バイト一致の強さは失われる。**

**router の判定は完全一致を要求できる。** expert 番号は整数で、argmax の同点以外では丸めで動かない。**「どのトークンがどの expert に行ったか」の完全一致を層ごとの関門にする** のが、この実装での一致検査のいちばん強い形になる。

## 形式 (2026-09-19 に `config.json` と transformers のソースで確認)

### `switch-base-8` の形

| | |
|---|---|
| `d_model` / `d_ff` / `d_kv` / `num_heads` | 768 / 3072 / 64 / 12 |
| encoder / decoder の層数 | 12 / 12 |
| **sparse な層** | `encoder_sparse_step` = `decoder_sparse_step` = 2 → **各 6 層**。残りは普通の dense FFN |
| `num_experts` / `num_selected_experts` | **8 / 1** (top-1) |
| `expert_capacity` | **64** |
| `router_bias` / `router_jitter_noise` | false / 0 |
| `router_dtype` | **float32** (fp16 で回すときもここは fp32) |
| `dense_act_fn` / `is_gated_act` | **relu / false** (T5 v1.1 の gated GELU ではない) |
| `layer_norm_epsilon` | 1e-6 |
| 相対位置 | `relative_attention_num_buckets` 32、`relative_attention_max_distance` 128 |
| 語彙 / pad / eos / `decoder_start_token_id` | 32128 / 0 / 1 / **0** |

**配布は `pytorch_model.bin` (1.24 GB) だけで、safetensors が無い。**`docs/idea.md` に書いた「HF の safetensors を自前で読む」経路がそのまま使えない。**pickle を Ruby から読む気は無いので、Python で 1 度変換する。**

**変換先は safetensors にした** (第 0 段階で決定)。`docs/idea.md` が用意していた「自前で読む 35 行」を実際に使うことになり、リポジトリの積み残しが 1 つ片付く。名前が残るので、transformers と層ごとに突き合わせるときにも都合がよい。形式の詳細は [docs/checkpoint-format-switch.md](../checkpoint-format-switch.md)。

## 1 トークンの計算 (transformers のソースを読んで確認)

### router (`SwitchTransformersTop1Router.forward`)

```
router_logits = hidden @ classifier          # [tokens, 8]、bias 無し
router_probs  = softmax(router_logits)       # fp32 で畳む
value, index  = max(router_probs, dim=-1)
one_hot       = one_hot(index, 8)
priority      = cumsum(one_hot, dim=-2)      # その expert の中で何番目に来たか (自分を含む)
mask          = priority <= expert_capacity
one_hot       = one_hot * mask               # 溢れたトークンは全 0 行になる
```

**`cumsum` が自分を含むので、先頭のトークンの priority は 1 である。**`<=` と合わせて、容量 64 なら 64 個目までが通る。

**溢れたトークンはどの expert にも行かない。**`SwitchTransformersLayerFF` が `hidden_states + dropout(forwarded_states)` なので、**残差だけで素通りする。**

### expert (`SwitchTransformersExperts.forward`)

```
final = zeros_like(hidden)
for expert_idx in 来たトークンが 1 つ以上ある expert:
    idx, top_x = where(expert_mask[expert_idx])
    y = expert(hidden[top_x]) * routing_weights[top_x, idx, None]
    final.index_add_(0, top_x, y)
```

**`where` と `index_add_` がデバイス上の添字である。** ここが cumo の塞がりに当たる。

**expert 本体は `wo(relu(wi(x)))` で bias 無し。** GPT-2 の FFN から活性が GELU → ReLU に変わっただけ。

### LayerNorm (`SwitchTransformersLayerNorm`)

**平均を引かない。**`variance = mean(x^2)` を **fp32 で畳み**、`x * rsqrt(variance + eps)` のあと `weight * x`。**Llama 2 の rmsnorm と同じ形だが、掛ける順序が `weight * (rstd * x)`** である (`mamba.c` は `x * weight * ss` だった。どちらに合わせるかは実測で決める)。

### decode では容量が効かない

**1 トークンずつ進める decoder では、1 層あたりのトークンが 1 個なので `expert_capacity` 64 に絶対に届かない。** 溢れが起きるのは **encoder 側で入力が 64 トークンを超えたとき** だけである。**関門に溢れを含めたいなら、encoder に長い入力を通すテストが要る。**

## コードの置き場

```
lib/narray_llm/models/switch/
  checkpoint.rb   形式の読み取り
  tokenizer.rb    SentencePiece
  model.rb        encoder / decoder / MoE
  router.rb       振り分け (2 つの書き方を切り替えられるようにする)
python/
  switch_torch.py  参照。中間活性と router の判定を落とす
  export_switch.py pytorch_model.bin からの変換
script/
  download_switch.rb
  switch_generate.rb
```

## 第 0 段階: 重みの取得とローダ — **完了** (2026-09-19)

### 作ったもの

- `script/download_switch.rb` — HF から 7 ファイル
- `python/export_switch.py` — pickle → fp32 の safetensors (437 本、2.48 GB)
- `lib/narray_llm/safetensors.rb` — 1 テンソルずつ seek して読む。**隙間と重なりをヘッダ段階で検査する**
- `lib/narray_llm/models/switch/checkpoint.rb` — `config.json` から **期待する 437 本の名前と形を組み立てて突き合わせる**

### 見積もりが 1 つ外れた

**「1.24 GB は 310M パラメータの fp32 に合う」と書いたが、外れた。** **現物は bf16 434 本 + fp32 6 本、693,361,920 パラメータ** (実体は 619,339,008)。`config.json` の `torch_dtype: bfloat16` のほうが正しく、**ファイルの大きさからの推測が間違っていた**。

### 現物で分かったこと

- **fp32 の 6 本は encoder の router だけ。** decoder の router は bf16 である。`router_dtype: float32` は **計算の型であって格納の型ではない**
- **埋め込みは 4 つの名前が 1 つの記憶域を指す** (`shared` / 両 `embed_tokens` / `lm_head`)。**Mamba の共有分類器と同じ話が、最初から 4 重で出てくる**
- **`relative_attention_bias` は各スタックの block 0 にしかない** (T5 は層で共有)
- **decoder の layer_norm は 3 本、encoder は 2 本。**`layer.M` の M がスタックで意味を変える
- **bf16 のまま置けない。** Numo に半精度が無いので、両バックエンドで読ませるなら変換時に fp32 へ広げるしかない。bf16 → fp32 は情報を落とさない

### 受け入れテスト — 7 件、両バックエンドで通る

- `config.json` の値がそのまま読めている
- **sparse な層が奇数ブロックで、そこにだけ router がある**
- **expert が 8 個ぶん別々に載っていて、中身が互いに違う**
- テンソル数 437 とパラメータ数 619,339,008 が `config.json` から組み立てた期待値と一致する
- **共有された 4 つの名前が同じ重みを答える**
- 相対位置バイアスが block 0 にしかない
- **`config.json` を偽って `num_experts` を 4 にすると読み込みが失敗する** (形の検査が効いていることの確認)

そして `python/export_switch.py` の出力は、**原本の 5 本と `torch.equal` でビット一致** する。

## 第一段階: encoder のフォワード 1 回 — **完了** (2026-09-20)

**decoder より先に encoder をやった。** cross-attention を後回しにでき、**`expert_capacity` が関わるのは encoder だけ** だからである。

### 作ったもの

- `python/switch_dump.py` — **forward hook で本物のモデルから採る。** 再実装ではないので、transformers の実際の計算とずれようがない。出力は safetensors で、Ruby 側は既存のリーダで読む
- `lib/narray_llm/models/switch/encoder.rb` — 12 ブロック。**振り分ける版と全部走らせる版の両方**

### `expert_capacity` は transformers では一度も効かない

**再現ケースで確かめた。**

```
one_hot の形             (200, 1, 8)  <- [tokens, 1, experts]
cumsum(dim=-2) は恒等か  True
priority の最大           1   (200 人が同じ expert に行っても)
容量 64 で落ちた数        0
```

`SwitchTransformersTop1Router.forward` の `torch.cumsum(expert_index, dim=-2)` が、**トークン軸ではなく大きさ 1 の軸** を畳んでいる。`token_priority == expert_index` なので最大が 1 で、`<= 64` は常に真になる。**本物のモデルでも、600 トークンで 1 つの expert に 222 個集まって、落ちたトークンは 0 だった。**

**参照に合わせる。**`expert_capacity:` は既定で `nil` (制限なし) にし、整数を渡したときだけ効くようにした。**論文と `config.json` は 64 と言うが、受け入れの相手は transformers である。**

### 関連位置バケットは、掛ける順序で答えが変わった

**距離がちょうど 64 のところで 1 バケットずれた。**

| | 計算 | 結果 |
|---|---|---|
| こちら (最初) | `log(d/8) * (8/log(16))` | 5.999999999999999 → 切り捨て 5 → bucket 13 |
| transformers | `log(d/8) / log(16) * 8` | 6.0 → bucket **14** |

**定数に畳んではいけない。** 128 トークンだと距離 64 の対が 64 組あり、そのぶん encoder 出力が `max|d|` 7.4e-02 ずれていた。**参照と同じ順序に直したらビット一致になった。**

### 受け入れテスト — 10 件、両バックエンドで通る

| | |
|---|---|
| encoder 出力が transformers と一致 | 6 トークンで 2.0e-06〜4.9e-06、128 トークンで 1.8e-05〜3.7e-05 (閾値 5e-05) |
| **関連位置バイアスがビット一致** | 6 と 128 の両方 |
| **router の判定が完全一致** | 6 層 x 2 つの書き方 x 2 つの長さ。**整数の判定なので誤差を許さない** |
| 2 つの書き方が同じ答え | Numo で **ビット一致**、Cumo で 2.7e-06 |
| **参照は 1 つも落とさない** | 1 つの expert に 80 個集まっても 0 |
| 容量を明示すると落ちる | 128 トークンで延べ 16 個 |

**Numo で 2 つの書き方がビット一致したのは予想していなかった。** 振り分ける版は expert ごとに行を集めてから掛け、全部走らせる版は全行に掛けてマスクする。**同じ積の並びになるので、BLAS が同じ順序で畳んでいるということだと思われるが、確かめていない。**

## 第二段階: decoder と生成 — **完了** (2026-09-20)

### 作ったもの

- `lib/narray_llm/models/switch/stack.rb` — encoder と decoder が共有するもの。**第一段階の encoder をここに割った**
- `decoder.rb` — self-attention (一方向)、cross-attention (位置バイアス無し)、`Cache`
- `model.rb` — encoder + decoder + 分類器と貪欲生成
- `python/switch_fixtures.py`、`script/switch_generate.rb`

### 分類器の前に `d_model ** -0.5` が掛かる

**重みが結ばれているときだけ掛かる** (`SwitchTransformersForConditionalGeneration.forward` の `if self.config.tie_word_embeddings`)。このチェックポイントは 4 つの名前が 1 つの記憶域なので、**掛かる側である**。忘れると当然トークン列が合わない。

### キャッシュは 2 種類ある

- **decoder の self-attention** — 1 行ずつ伸びる。既存の `KVCache` がそのまま使えた
- **cross-attention** — encoder の出力から 1 度だけ射影して、その後は動かない

### decoder の相対位置バイアスは一方向

`bidirectional` が false なので **バケットが半分に割られず 32 のまま** で、`max_exact` が 16 になる。そして **先の位置は 0 に畳まれる**。1 行だけ問い合わせるので `offset` に現在位置を渡す。

### 受け入れテスト — 7 件、両バックエンドで通る

**3 つのプロンプト x 2 つの書き方 x 2 バックエンドで、トークン列が完全一致した。**

| プロンプト | 長さ | 結果 |
|---|---|---|
| short | 6 | 25 トークン、予算いっぱい |
| sentinel | 10 | **6 トークン、eos で停止** |
| long | 128 | 25 トークン。**1 つの expert に容量超の 80 個が集まる入力** |

ほかに、`decoder_start_token_id` から始まること、2 度目が 1 度目と同じであること、範囲外のトークン id を撥ねること、**decoder のバケットが先の位置を 0 に畳むこと**。

## 第三段階: 計測 — **完了** (2026-09-20)

**答えは「振り分けるほうが速い」だった。** 同期を 7 回から 1 回に減らした側が、2 条件とも負けた。

| | 同期/token | カーネル/token | decode 72 |
|---|---|---|---|
| (a) 振り分ける | **7** | **1821** | **149.81** |
| (b) 全部走らせる | **1** | 2121 | 138.88 |
| ペア比 (a)/(b) | | | **1.076 (1.065〜1.101) 10/10** |

対照は 2 条件とも 1 をまたぐ。

**そして 3 実装とも同じ向きだった。** CuPy 1.020、PyTorch 1.120、いずれも 10/10 でまたがない。**「振り分けるほうが速い」は cumo の同期の性質ではなく、この形の性質である。**

**3 実装の比較では条件で像が変わる。** encode 2048 では 3 つに差が無く (すべて 1 をまたぐ)、decode 72 では **PyTorch が 24% 速く、CuPy が 2.6 倍遅い**。詳細は [docs/results/switch-base-8.md](../results/switch-base-8.md)。

**差を同期に帰することはできない。**(a) と (b) は同期の回数だけでなくカーネル数も仕事量も違う。**言えるのは「この形では、同期を 6 回節約するより 8 倍の仕事を避けるほうが効く」まで。**

### そのあとで 3 つ動いた (2026-09-20)

**codex のレビューで 3 件直した。** 同点の expert を全部選んでいたこと、`DTYPE=fp16/bf16` で重みをバイト列として読み替えていたこと、重みが無いときにテストがスキップにならなかったこと。2 つ目を直す途中で **fp16 ではこのモデルが走らない** ことも分かった (残差が 5 ブロック目で Inf)。**bf16 は走ってトークン列も変わらない。**

**head ごとのコピーをやめて 2637 → 1821 本。** decode で **1.300 (1.235〜1.352) 10/10**、encode 2048 では **動かない** (1.003、またぐ)。**3 実装の表が裏返り、PyTorch が 24% 勝っていたのが Cumo が 2.9% 勝つ形になった** (Python の 2 つにも同じ形を入れてから測り直した)。

**`cumsum` をやめると 6.4% 速い。** cumo の `cumsum` は 8192 要素未満でホストに落ちる (ソースに `FIXME`)。三角行列との積に替えると消える。**カーネル数 2121 本も答えも同じ。**

**それを「同期 1 回 74 us」と書いたが、取り下げた。** codex の指摘で対照を取った — **同じ場所で同期だけを 6 回足すアーム** は **1.007 で 1 をまたぐ**。**同期はこの地点では測れる大きさではなかった。** **残る費用の内訳は未分離である** (ホストループは候補だが測っていない)。**本数と出力が同じでも、違いが同期だけとは言えない。**

## 未確認

- **SentencePiece (`spiece.model`) をどう読むか。** protobuf なので、既存の 3 つのトークナイザとは別物
- **bf16 のまま置いて Ruby 側で広げる手。** bf16 は fp32 の上位 16 ビットなので、`UInt16` で読んで 16 ビット左シフトすれば型が無くても作れる。**ファイルが半分になる** が、第 0 段階では踏み込まなかった
- **相対位置バイアスの bucket 表**。transformers の `_relative_position_bucket` を読んでいない
- **Numo で 2 つの書き方がビット一致した理由。** Cumo では 2.7e-06 ずれる
- **`expert_capacity` を transformers に報告するか。** 再現ケースはあるが、こちらから出していない
- **(a) と (b) のどちらが速いか。** 測っていない
