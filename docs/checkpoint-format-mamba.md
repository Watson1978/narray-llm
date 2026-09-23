# Mamba のチェックポイント形式

[kroggen/mamba.c](https://github.com/kroggen/mamba.c) の `learning` ブランチ (既定、Mamba 1) が読む version 1 形式。出典は `export.py` の `model_export` と `mamba.c` の `load_model_file` / `memory_map_weights`。

## 作り方

配布されている `.bin` は無い。**HuggingFace の PyTorch 重みを `export.py` が変換する。**

```
ruby script/download_mamba.rb
cd vendor/mamba.c && ../../python/.venv/bin/python export.py \
  ../../data/mamba-130m ../../data/mamba-130m.bin
```

`export.py` は `state-spaces/mamba-...` という名前を渡すと `transformers` のキャッシュに落とすが、**このリポジトリは `config.json` と `pytorch_model.bin` を `data/mamba-130m/` に直接取ってディレクトリを渡す**。`load_model` が `os.path.isdir` を見るのでそのまま通り、重みが他の `.bin` と同じ場所に残る。

**参照実装は `make` で建てる。`make fast` を使わないこと。**`fast` は `-Ofast` (= `-ffast-math`) で、ビット一致の参照にならない。

## ヘッダ

| バイト | 内容 |
|---|---|
| 0〜3 | マジック `0x4d616d62` (`"Mamb"`)、uint32 |
| 4〜7 | バージョン `1`、int32 |
| 8〜39 | int32 が 8 個 |
| 40〜255 | 0 埋め |

整数 8 個の順序は `n_layers`, `vocab_size`, `d_model`, `d_inner`, `dt_rank`, `d_state`, `d_conv`, `shared_classifier`。

**`mamba.c` は `fread(config, sizeof(Config), 1, file)` で 9 個ぶん読む** (`mamba.c:185`)。9 個目の `rounded_vocab_size` はファイルには無く 0 埋めから読まれ、直後に計算で上書きされる。

## `rounded_vocab_size`

**`vocab_size` が 8 の倍数でなければ 8 の倍数に切り上げ、埋め込みと分類器はその大きさで格納される** (`mamba.c:187`)。

```
vocab_size % 8 != 0 なら rounded = vocab_size + (8 - vocab_size % 8)
```

mamba-130m は `vocab_size` が **50277** で、格納は **50280** 行。トークナイザの表は 50277 個なので、**末尾 3 行はどのトークンにも対応しない**。

## テンソルの並び

`memory_map_weights` (`mamba.c:152`) がポインタを進める順そのまま。**テンソル名ごとに全層をまとめて書く** (llama2.c と同じ)。dtype はすべて fp32。

| | 形 |
|---|---|
| `embedding` | `[rounded_vocab, d_model]` |
| `in_proj` | `[n_layers, 2 * d_inner, d_model]` |
| `conv1d_weight` | `[n_layers, d_inner, d_conv]` |
| `conv1d_bias` | `[n_layers, d_inner]` |
| `x_proj` | `[n_layers, dt_rank + 2 * d_state, d_inner]` |
| `dt_proj_weight` | `[n_layers, d_inner, dt_rank]` |
| `dt_proj_bias` | `[n_layers, d_inner]` |
| `A` | `[n_layers, d_inner, d_state]` |
| `D` | `[n_layers, d_inner]` |
| `out_proj` | `[n_layers, d_model, d_inner]` |
| `norm` | `[n_layers, d_model]` |
| `final_norm` | `[d_model]` |
| `lm_head` | `[rounded_vocab, d_model]` — `shared_classifier` が 0 のときだけ |

**`A` は `A_log` ではない。**`export.py` が `A = -exp(A_log)` に変換して書くので、読む側は変換しない。**全要素が負** である。

**`conv1d_weight` は torch では `[d_inner, 1, d_conv]`** だが、真ん中の軸が 1 なので要素数は `d_inner * d_conv`。`mamba.c` も `d_inner * 1 * d_conv` として扱う。

## mamba-130m の値

| | |
|---|---|
| `n_layers` | 24 |
| `vocab_size` | 50277 (格納は 50280) |
| `d_model` | 768 |
| `d_inner` | 1536 |
| `dt_rank` | 48 |
| `d_state` | 16 |
| `d_conv` | 4 |
| `shared_classifier` | 1 |

ファイルは **516,541,696 バイト** で、`256 + 4 x パラメタ数` と一致する。

**並びは元の PyTorch と突き合わせて検証した。**`embedding`、`in_proj[0]`、`conv1d_weight[0]`、`x_proj[23]`、`A[0]`、`A[23]`、`out_proj[23]`、`final_norm` の 8 つが **すべてビット一致** する。最終層が合うので層ごとのストライドも正しい。

## 小さいチェックポイント

**公式の最小が 130m しかない。** stories260K に当たるものが無いので、`script/mamba_tiny.rb` が同じ形式で小さいものを書く。固定 seed なので **同じコマンドが同じバイトを出す**。

```
ruby script/mamba_tiny.rb          # data/mamba_tiny.bin、69,760 バイト
```

既定は `n_layers=2, vocab=64, d_model=32, d_inner=64, dt_rank=2, d_state=4, d_conv=4`。`A` は負で書く。**`mamba.c` がこのファイルの config を正しく読むことは確認済み。**
