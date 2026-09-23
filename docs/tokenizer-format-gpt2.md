# llm.c トークナイザ (gpt2_tokenizer.bin) の形式

第 0 段階の [checkpoint-format-gpt2.md](checkpoint-format-gpt2.md) と同じ流儀で、推測せず llm.c のソースを読んで確定させたもの。出典は `karpathy/llm.c` の master (2026-08-21 取得) で、行番号は取得時点のもの。

- `llmc/tokenizer.h` — 読み出し側。`tokenizer_init` (:41)、`tokenizer_decode` (:86)、`safe_printf` (:19)
- `train_gpt2.py` — 書き出し側。`write_tokenizer` (:509)
- `train_gpt2.c` — 利用側。EOT からの生成開始 (:1127-1129)、デコードと表示 (:1151-1152)

すべて little-endian。

## 取得元

`dev/download_starter_pack.sh` の `BASE_URL` に `gpt2_tokenizer.bin` を足したもの。`rake download:gpt2` が取得する 3 ファイルのうちの 1 つ。実ファイルは 372,108 バイト。

## ヘッダ: uint32 x 256 (1024 バイト)

`tokenizer.h:54-55` が `uint32_t header[256]` を丸ごと読み、`:56-69` で解釈する。**checkpoint と違ってここは符号なし 32bit** なので、読み出しは `unpack('L<*')` を使う。

| index | 意味 | gpt2_tokenizer.bin での値 | 出典 |
|---|---|---|---|
| 0 | magic | 20240328 | `tokenizer.h:56` |
| 1 | version | 2 | `tokenizer.h:57`, `train_gpt2.py:513` |
| 2 | `vocab_size` | 50257 | `tokenizer.h:58` |
| 3 | `eot_token` | 50256 (version 2 のみ) | `tokenizer.h:65` |
| 4..255 | 未使用 (0 埋め) | — | `train_gpt2.py:511` が `torch.zeros(256)` で作る |

`vocab_size` は書き出し側で `enc.max_token_value + 1` (`train_gpt2.py:510`)。つまり **padding 無しの V = 50257** で、checkpoint の `wte` が持つ Vp = 50304 とは別物。

## 本体 (`tokenizer.h:70-80`)

ヘッダ直後から、`vocab_size` 個ぶん次を繰り返す。

| 型 | 意味 |
|---|---|
| `uint8` | `length` — 続くバイト列の長さ |
| `length` バイト | トークンの生バイト列 |

- `length` は 1 以上 (`tokenizer.h:75` が `assert(length > 0)`)。
- `length` は 255 以下 (`train_gpt2.py:521` が `assert length < 256`)。
- 最初のトークン (id=0) は 1 バイトの `0x21` = `!`。GPT-2 のバイトレベル BPE の語彙順そのまま。

C 側は読んだあとに NUL 終端を足している (`tokenizer.h:78`) が、これは `printf` で表示するための都合であってファイル上のデータではない。**トークンのバイト列には 0x00 を含みうる** ので、Ruby 側では長さで扱い、NUL 終端に頼らない。

## EOT トークン id をハードコードしないこと

`tokenizer.h:59-69` が version ごとに分岐している。

- version 1: EOT のフィールドが無いので llm.c が 50256 を決め打ちする。ただし `assert(tokenizer->vocab_size == 50257)` で防御している (`tokenizer.h:62`)
- version 2: `header[3]` に入っている (`tokenizer.h:65`)。書き出し側は tiktoken の `enc.eot_token` (`train_gpt2.py:514`)
- それ以外: llm.c は `exit(EXIT_FAILURE)` する (`tokenizer.h:67`)

したがって 50256 はファイルから読む値であって、定数として書いてよいのは **version 1 のフォールバックとしてだけ**。

llm.c は無条件生成の開始トークンとして EOT を使う (`train_gpt2.c:1127-1129` の `// fill up gen_tokens with the GPT2_EOT, which kicks off the generation`)。このリポジトリの `script/gpt2_generate.rb` の既定もこれに合わせてある。

## デコードはバイト列の連結

`tokenizer_decode` (`tokenizer.h:86-95`) は id を範囲検査して表を引くだけ。`token_id < vocab_size` でなければ「invalid token id」と表示して NULL を返す。

GPT-2 はバイトレベル BPE なので、**トークン境界は UTF-8 の文字境界と一致しない**。1 文字が複数トークンに割れることがあり、途中まで連結した時点では不正な UTF-8 になる。したがって:

1. トークンのバイト列をすべて連結してから
2. `force_encoding('UTF-8')` し
3. 不正シーケンスは `scrub` で U+FFFD に置換する

llm.c の `safe_printf` (`tokenizer.h:19-38`) は、1 バイトトークンのうち印字可能でも空白でもないものを **表示しない** という別の対処をしている。これは端末に制御コードを吐かないための表示側の都合であって形式の一部ではないので、このリポジトリでは真似せず、バイトは保持したうえで UTF-8 として不正な部分だけ置換する方針を採る。

## エンコード (BPE) は実装しない

PLAN-gpt2.md の方針どおり、この段階で作るのはデコードのみ。プロンプトはトークン化済みの id 列で渡す。`script/gpt2_generate.rb` の既定が EOT 1 個からの無条件生成なのは、エンコーダ無しで動かせるようにするため。
