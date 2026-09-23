# llm.c チェックポイント形式

このドキュメントは推測ではなく llm.c のソースを読んで確定させたもの。出典は `karpathy/llm.c` の master (2026-08-21 取得) の以下のファイル。行番号は取得時点のもの。

- `train_gpt2.c` — `gpt2_build_from_checkpoint` (:707), `ParameterTensors` (:537), `fill_in_parameter_sizes` (:556), `malloc_and_point_parameters` (:580)
- `test_gpt2.c` — debug state の読み出し (:53-:81)
- `train_gpt2.py` — 書き出し側。`write_tensors` (:395), `pad_vocab` (:429), `write_model` (:449), `write_state` (:479), `write_tokenizer` (:509)
- `llmc/tokenizer.h` — `tokenizer_init` (:40)
- `dev/download_starter_pack.sh` — 取得元 URL

すべて little-endian、C の `int` は int32、浮動小数点は fp32。

## 取得元

`dev/download_starter_pack.sh` の `BASE_URL`:

```
https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/<name>?download=true
```

第 0 段階で使うのは `gpt2_124M.bin`、`gpt2_124M_debug_state.bin`、`gpt2_tokenizer.bin` の 3 つ。

## 1. gpt2_124M.bin (モデル重み)

### ヘッダ: int32 x 256 (1024 バイト)

`train_gpt2.c:711-712` が `int model_header[256]` を丸ごと読み、`:713-727` で解釈する。

| index | 意味 | GPT-2 124M での値 | 出典 |
|---|---|---|---|
| 0 | magic | 20240326 | `train_gpt2.c:713` |
| 1 | version | 3 (fp32 + padded vocab) | `train_gpt2.c:714`, `train_gpt2.py:453-455` |
| 2 | `max_seq_len` (maxT) | 1024 | `train_gpt2.c:722` |
| 3 | `vocab_size` (V) | 50257 | `train_gpt2.c:723` |
| 4 | `num_layers` (L) | 12 | `train_gpt2.c:724` |
| 5 | `num_heads` (NH) | 12 | `train_gpt2.c:725` |
| 6 | `channels` (C) | 768 | `train_gpt2.c:726` |
| 7 | `padded_vocab_size` (Vp) | 50304 | `train_gpt2.c:727` |
| 8..255 | 未使用 (0 埋め) | — | `train_gpt2.py:456` が `torch.zeros(256)` で作る |

version 3 以外は llm.c が拒否する (`train_gpt2.c:714`)。bf16 版は version 5 で、これは別ファイル (`gpt2_124M_bf16.bin`) なのでこのローダは扱わない。

### パラメータ本体

ヘッダ直後から fp32 が隙間なく並ぶ (`train_gpt2.c:749` が `num_parameters` 個を一度に読む)。並び順は `malloc_and_point_parameters` (`train_gpt2.c:588-592`) のポインタ配列の順で、書き出し側 `train_gpt2.py:395-425` の `write_tensors` と一致する。

要素数は `fill_in_parameter_sizes` (`train_gpt2.c:556-577`) がすべて。

| # | 名前 | shape | 要素数の式 | 124M での要素数 |
|---|---|---|---|---|
| 0 | `wte` | `[Vp, C]` | `Vp*C` | 38,633,472 |
| 1 | `wpe` | `[maxT, C]` | `maxT*C` | 786,432 |
| 2 | `ln1w` | `[L, C]` | `L*C` | 9,216 |
| 3 | `ln1b` | `[L, C]` | `L*C` | 9,216 |
| 4 | `qkvw` | `[L, 3C, C]` | `L*3C*C` | 21,233,664 |
| 5 | `qkvb` | `[L, 3C]` | `L*3C` | 27,648 |
| 6 | `attprojw` | `[L, C, C]` | `L*C*C` | 7,077,888 |
| 7 | `attprojb` | `[L, C]` | `L*C` | 9,216 |
| 8 | `ln2w` | `[L, C]` | `L*C` | 9,216 |
| 9 | `ln2b` | `[L, C]` | `L*C` | 9,216 |
| 10 | `fcw` | `[L, 4C, C]` | `L*4C*C` | 28,311,552 |
| 11 | `fcb` | `[L, 4C]` | `L*4C` | 36,864 |
| 12 | `fcprojw` | `[L, C, 4C]` | `L*C*4C` | 28,311,552 |
| 13 | `fcprojb` | `[L, C]` | `L*C` | 9,216 |
| 14 | `lnfw` | `[C]` | `C` | 768 |
| 15 | `lnfb` | `[C]` | `C` | 768 |

合計 `num_parameters` = **124,475,904**。ファイルサイズ = `1024 + 4 * 124,475,904` = **497,904,640** バイト。

### 落とし穴 3 つ

**(a) `wte` は V ではなく Vp 行。** `ParameterTensors` のコメント (`train_gpt2.c:538`) は `// (V, C)` と書いてあるが、実際に確保・読み込みされるのは `fill_in_parameter_sizes` の `param_sizes[0] = Vp * C` (`train_gpt2.c:561`) で **Vp 行**。コメントの方が古い。ここを V で読むと以降の全テンソルのオフセットがずれる。

**(b) 埋めた行 (V..Vp-1) は厳密に 0.0。** `train_gpt2.py:429` の `pad_vocab(tensor, multiple=128, value=0)` が `F.pad(..., value=value)` で 0 埋めする。つまり `wte[50257..50303, :]` は全部 0.0。オフセットが正しいことの強い検査になる。logits を取るときはこの行を捨てる必要がある (`test_gpt2.c:120` の `for (int v = 0; v < V; v++) // note we only loop to V (ignoring padding)`)。

**(c) 重み行列は PyTorch の `nn.Linear` 配置、つまり `[out, in]`。** HuggingFace の GPT-2 は `Conv1D` で `[in, out]` を持つが、`train_gpt2.py:223-232` が

```python
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
... sd[k].copy_(sd_hf[k].t())
```

で転置してから書き出している。したがってファイル上の `qkvw[l]` は `[3C, C]` = `[out, in]`。フォワードでは `y = x . W^T + b` になる (第一段階で効いてくる)。

## 2. gpt2_124M_debug_state.bin (参照入力・参照出力)

`test_gpt2.c:53-82` がすべて。

### ヘッダ: int32 x 256 (1024 バイト)

| index | 意味 | 値 | 出典 |
|---|---|---|---|
| 0 | magic | 20240327 | `test_gpt2.c:56` |
| 1 | version | 2 (padded vocab 対応) | `test_gpt2.c:57`, `train_gpt2.py:485` |
| 2 | B (batch size) | 4 | `test_gpt2.c:62` |
| 3 | T (sequence length) | 64 | `test_gpt2.c:63` |
| 4..255 | 未使用 (0 埋め) | — | `train_gpt2.py:483` |

### ヘッダ直後の並び (`test_gpt2.c:78-82`)

| # | 名前 | 型 | 個数 | 124M/B=4,T=64 でのバイト数 |
|---|---|---|---|---|
| 0 | `x` (入力トークン) | int32 | `B*T` | 1,024 |
| 1 | `y` (ターゲット) | int32 | `B*T` | 1,024 |
| 2 | `expected_logits` | fp32 | `B*T*V` | 51,463,168 |
| 3 | `expected_loss` | fp32 | 1 | 4 |
| 4 | `expected_grads` | fp32 | `num_parameters` | 497,903,616 |

合計ファイルサイズ = **549,369,860** バイト。

**`expected_logits` は Vp ではなく V (50257) 幅。** `test_gpt2.c:74` が `B*T*V` で確保し、`:125` が `expected_logits[bt*V + v]` で読む一方、C 側の計算結果は `calculated_logits[bt*Vp + v]` (`:121`) で引く。参照 logits は padding 無しなので、比較するときは自分の logits 側を V 列に切り詰める。

**`expected_grads` の並びは重みと同一** (`test_gpt2.c:69` が `malloc_and_point_parameters(&expected_grads, model.param_sizes)` を同じ `param_sizes` で呼ぶ)。`wte` の勾配も `pad_vocab(..., value=0)` されている (`train_gpt2.py:491`)。第 0〜3 段階は推論のみなので読み飛ばすが、オフセット計算には必要。

### 外部から検証できる値

`test_gpt2.c:89-90` の `expected_losses[0] = 5.270007133483887f` は、この `x`/`y` に対する PyTorch の loss。`test_gpt2.c:141` が `expected_loss` (ファイル内の値) を `model.mean_loss` と `1e-2` で突き合わせているので、ファイル内の `expected_loss` はこの値と `1e-2` 以内で一致する。debug state のオフセットが正しいかの外部アンカーとして使える。

## 3. gpt2_tokenizer.bin (デコード用テーブル)

`llmc/tokenizer.h:53-84`。第二段階で使うので形式だけ控えておく。

### ヘッダ: uint32 x 256 (1024 バイト)

| index | 意味 | 値 |
|---|---|---|
| 0 | magic | 20240328 (`tokenizer.h:56`) |
| 1 | version | 2 (`tokenizer.h:64`) |
| 2 | `vocab_size` | 50257 (`tokenizer.h:58`) |
| 3 | `eot_token` | 50256 (version 2 のみ。`tokenizer.h:65`) |

### 本体 (`tokenizer.h:70-80`)

`vocab_size` 個ぶん、次を繰り返す:

- `uint8 length` (1 バイト、必ず 1 以上)
- `length` バイトの生バイト列 (UTF-8 とは限らないので String は ASCII-8BIT で保持する)

## 参照値の出所について

このリポジトリのテストが使う「先頭数要素の参照値」は、llm.c の出力ではなく HuggingFace の `openai-community/gpt2` の `model.safetensors` (dtype F32) から safetensors ヘッダのオフセットを使って直接読んだもの。`train_gpt2.py:216` の `GPT2LMHeadModel.from_pretrained(model_type)` が読むのと同じ重みなので、fp32 でビット一致するはず、という前提で完全一致を要求している。

対応関係 (`train_gpt2.py:399-425` の名前対応):

| このリポジトリ | HuggingFace safetensors | 備考 |
|---|---|---|
| `wte` | `wte.weight` | 先頭 V 行のみ。V.. は 0 padding |
| `wpe` | `wpe.weight` | |
| `ln1w`/`ln1b`[l] | `h.{l}.ln_1.weight`/`.bias` | |
| `qkvw`/`qkvb`[l] | `h.{l}.attn.c_attn.weight`/`.bias` | weight は **転置** されている |
| `attprojw`/`attprojb`[l] | `h.{l}.attn.c_proj.weight`/`.bias` | weight は **転置** |
| `ln2w`/`ln2b`[l] | `h.{l}.ln_2.weight`/`.bias` | |
| `fcw`/`fcb`[l] | `h.{l}.mlp.c_fc.weight`/`.bias` | weight は **転置** |
| `fcprojw`/`fcprojb`[l] | `h.{l}.mlp.c_proj.weight`/`.bias` | weight は **転置** |
| `lnfw`/`lnfb` | `ln_f.weight`/`.bias` | |
