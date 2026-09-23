# llama2.c チェックポイント形式 (legacy / v0)

このドキュメントは推測ではなく llama2.c のソースを読んで確定させたもの。出典は `karpathy/llama2.c` の master (2026-09-16 取得) の以下のファイル。行番号は取得時点のもの。

- `run.c` — `Config` (:19), `memory_map_weights` (:110), `read_checkpoint` (:142), `matmul` (:217), `forward` (:231), `build_tokenizer` (:385)
- `export.py` — 書き出し側。`legacy_export` (:75)
- `test_all.py` — stories260K の取得先と、温度 0 の既知出力
- `README.md` — tinyllamas のモデル表 (:148-155)

すべて little-endian、C の `int` は int32、浮動小数点は fp32。

**GPT-2 (llm.c) と違い、マジックナンバーもバージョンも無い。** 先頭にいきなり `Config` が来る。`export.py` には magic 付きの v1 / v2 形式もあるが、tinyllamas で配られている `.bin` は legacy (v0) なのでこのローダは v0 だけを扱う。

## 取得元

| ファイル | サイズ | 取得元 |
|---|---|---|
| `stories260K.bin` | 1,056,540 B | `huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin` |
| `tok512.bin` | 6,227 B | 同上の `stories260K/tok512.bin` (語彙 512) |
| `stories110M.bin` | 438,381,596 B | `huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin` |
| `tokenizer.bin` | 433,869 B | llama2.c リポジトリの `tokenizer.bin` (語彙 32000) |

260K の 4 ファイルを名指ししているのは `test_all.py:30-32`。大きいモデルの取得先は README のモデル表。

## ヘッダ: int32 x 7 (28 バイト)

`run.c:145` が `fread(config, sizeof(Config), 1, file)` で `Config` を丸ごと読む。`Config` は `run.c:19-27` の 7 つの int。

| index | フィールド | 意味 |
|---|---|---|
| 0 | `dim` | 埋め込み次元 |
| 1 | `hidden_dim` | FFN の中間次元 |
| 2 | `n_layers` | 層数 |
| 3 | `n_heads` | クエリヘッド数 |
| 4 | `n_kv_heads` | キー/バリューのヘッド数 (`< n_heads` なら multiquery) |
| 5 | `vocab_size` | 語彙数。**符号が分類器共有のフラグ** (下記) |
| 6 | `seq_len` | 最大系列長 |

派生する量 (`run.c:111`, `:236-237`):

```
head_size = dim / n_heads
kv_dim    = dim * n_kv_heads / n_heads
kv_mul    = n_heads / n_kv_heads
```

### vocab_size の符号が分類器共有のフラグ

`run.c:147-149`:

```c
int shared_weights = config->vocab_size > 0 ? 1 : 0;
config->vocab_size = abs(config->vocab_size);
```

**負なら分類器を共有しない** ので、末尾に `wcls` が別途入る。正なら `wcls` は `token_embedding_table` を指すだけで、ファイルには 1 つしか入っていない (`run.c:139`)。書き出し側も同じ約束で、`export.py:84-85` が共有でない場合に `p.vocab_size` を負にしている。コメントが `bit yikes` と言っているとおりの仕掛けである。

**配布されている stories260K と stories110M はどちらも共有 (正)** なので、負の経路は実ファイルでは踏めない。テストは仕様から組み立てた合成ファイルで押さえている。

## パラメータ本体

ヘッダ直後から fp32 が隙間なく並ぶ。並び順と各テンソルの広がりは `memory_map_weights` (`run.c:110-140`) のポインタの進め方がすべて。書き出し側 `legacy_export` (`export.py:91-123`) と一致する。

| # | 名前 | shape (row-major) | 要素数 |
|---|---|---|---|
| 1 | `token_embedding_table` | `[vocab_size, dim]` | `vocab_size * dim` |
| 2 | `rms_att_weight` | `[n_layers, dim]` | `n_layers * dim` |
| 3 | `wq` | `[n_layers, n_heads * head_size, dim]` | `n_layers * dim * (n_heads * head_size)` |
| 4 | `wk` | `[n_layers, kv_dim, dim]` | `n_layers * dim * kv_dim` |
| 5 | `wv` | `[n_layers, kv_dim, dim]` | `n_layers * dim * kv_dim` |
| 6 | `wo` | `[n_layers, dim, n_heads * head_size]` | `n_layers * (n_heads * head_size) * dim` |
| 7 | `rms_ffn_weight` | `[n_layers, dim]` | `n_layers * dim` |
| 8 | `w1` | `[n_layers, hidden_dim, dim]` | `n_layers * dim * hidden_dim` |
| 9 | `w2` | `[n_layers, dim, hidden_dim]` | `n_layers * hidden_dim * dim` |
| 10 | `w3` | `[n_layers, hidden_dim, dim]` | `n_layers * dim * hidden_dim` |
| 11 | `rms_final_weight` | `[dim]` | `dim` |
| — | (読み飛ばし) `freq_cis_real` | — | `seq_len * head_size / 2` |
| — | (読み飛ばし) `freq_cis_imag` | — | `seq_len * head_size / 2` |
| 12 | `wcls` | `[vocab_size, dim]` | 共有なら 0、非共有なら `vocab_size * dim` |

### shape の向きは「出力が先」

`run.c` の構造体のコメントは `wq` を `(layer, dim, n_heads * head_size)` と書いているが、**実際の並びは逆で出力が先** である。根拠は 2 つ。

1. `matmul(xout, x, w, n, d)` は `w[i * n + j]` と読む (`run.c:217-229`)。つまり `w` は `(d, n)` の row-major。`wk` の呼び出しは `matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim)` (`run.c:262`) なので `n = dim`、`d = kv_dim`、shape は `[kv_dim, dim]`。
2. `export.py:101` が書き出すのは `layer.attention.wk.weight` そのもので、torch の `nn.Linear(dim, n_kv_heads * head_size)` の `weight` は `(out_features, in_features)` = `(kv_dim, dim)`。

`wq` と `wo` は両辺が `dim` なので見分けが付かないが、`wk` / `wv` で向きが決まる。`w1` / `w2` / `w3` については構造体のコメントのほうが正しい (`w1` が `(layer, hidden_dim, dim)`)。**コメントの順序は信用できないので、`matmul` の引数から決めること。**

### freq_cis の読み飛ばし

`run.c:136-137` が `seq_len * head_size / 2` 個ずつ 2 回進める。RoPE の表を事前計算していた頃の名残で、現在の `run.c` は使わず毎回その場で計算する (`run.c:265-280`)。**ファイルには存在するので、読み飛ばさないと `wcls` の位置がずれる。**

割り算は C の整数演算で `(seq_len * head_size) / 2` である。`export.py:118-119` が書くのは `freqs_cos[:max_seq_len]` で shape が `(seq_len, head_size / 2)` なので、`head_size` が偶数なら同じ値になる。

**読み飛ばすときも読むこと。**`seek` で飛ばすと、この領域で切り詰められたファイルが EOF 検査を素通りする (共有分類器の場合、後ろに読むものが無いため)。

## 実ファイルの値

| | dim | hidden_dim | n_layers | n_heads | n_kv_heads | vocab_size | seq_len | 分類器 |
|---|---|---|---|---|---|---|---|---|
| stories260K | 64 | 172 | 5 | 8 | **4** | 512 | 512 | 共有 |
| stories110M | 768 | 2048 | 12 | 12 | 12 | 32000 | 1024 | 共有 |

**stories260K は GQA である** (`n_kv_heads` 4 < `n_heads` 8、`kv_mul` = 2)。README のモデル表がそう書いており、ヘッダの実測とも一致する。**形式検証用に選んだ最小のモデルが multiquery なので、GQA は後回しにできない。** 110M のほうは `n_heads == n_kv_heads` で素の MHA。

`hidden_dim` は 172 で、32 の倍数ではない。ヘッダから読むので問題にはならないが、`hidden_dim` を `dim` から計算で出そうとしてはいけない。

要素数の合計は 264,128 と 109,595,392 で、`28 + 4 * (合計 + freq_cis)` が **両方ともファイルサイズに 1 バイトの差もなく一致する**。これが形式を読み違えていないことの検査になっている (test/test_llama2_checkpoint.rb)。

## このリポジトリでの読み方

`NArrayLLM::Llama2::Checkpoint` が上の順で `from_binary` + `reshape` に割り付ける。`wcls` は共有のとき `token_embedding_table` と同じオブジェクトを指す (`run.c` と同じ約束)。`Config` は `head_size` / `kv_dim` / `kv_mul` / `grouped_query?` を派生させて持つ。

ヘッダにマジックが無いので、**壊れたファイルを形式の段階で弾く手段が無い**。代わりに次を検査する。

- `dim` などが正であること、`dim` が `n_heads` で割り切れること、`n_heads` が `n_kv_heads` で割り切れること
- 全テンソルを読み切ったあとがちょうど EOF であること (前後どちらにずれても `FormatError`)
