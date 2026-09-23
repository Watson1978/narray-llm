# llama2.c トークナイザ形式

このドキュメントは推測ではなく llama2.c のソースを読んで確定させたもの。出典は `karpathy/llama2.c` の master (2026-09-16 取得) の以下。行番号は取得時点のもの。

- `run.c` — `Tokenizer` (:377), `build_tokenizer` (:385), `decode` (:418), `safe_printf` (:431), `generate` (:729)
- `tokenizer.py` — 書き出し側 (`export` :34)
- `test_all.py` — 温度 0 の既知出力

すべて little-endian、`int` は int32、`float` は fp32。

**GPT-2 (llm.c) と違い、マジックナンバーもバージョンも語彙数も入っていない。**`run.c:387` のコメントが `i should have written the vocab_size into the tokenizer file... sigh` と言っているとおりで、**語彙数はモデルの `Config` から渡す**。このリポジトリの `Llama2::Tokenizer.load` が `vocab_size:` を要求するのはこのため。

## 取得元

| ファイル | サイズ | 語彙 | 対象 |
|---|---|---|---|
| `tokenizer.bin` | 433,869 B | 32000 | stories15M / 42M / 110M、Llama 2 本体 |
| `tok512.bin` | 6,227 B | 512 | stories260K 専用 |

`tokenizer.bin` は llama2.c リポジトリ直下、`tok512.bin` は tinyllamas の `stories260K/` にある。

## 形式

`build_tokenizer` (`run.c:396-408`) がそのまま仕様になる。

```
int32   max_token_length        1 個
繰り返し vocab_size 回:
  float32 score                 SentencePiece のスコア (マージの優先度)
  int32   len                   バイト数
  bytes   piece[len]            NUL 終端されていない
```

ファイルはこれで終わる。**このリポジトリは `4 + vocab_size * 8 + Σlen` がファイルサイズと一致することを検査する** (両ファイルとも一致)。`max_token_length` は `tokenizer.bin` が 27、`tok512.bin` が 7。

`score` はエンコード側 (`run.c:452` 以降の BPE マージ) でしか使わない。**このリポジトリはエンコードを実装しない** ので読み込むだけ。

### 先頭の 3 つは固定

| id | piece | 意味 |
|---|---|---|
| 0 | `<unk>` | 未知語 |
| 1 | `\n<s>\n` | BOS |
| 2 | `\n</s>\n` | EOS |
| 3..258 | `<0x00>` .. `<0xFF>` | 生バイト |

`tokenizer.bin` と `tok512.bin` で同じ並びだった。

## デコードの規則

`decode` (`run.c:418-429`) は 2 つだけ特別扱いをする。**どちらも直前のトークンを見る必要がある** ので、`decode` は 1 トークンでは閉じない。

1. **BOS の直後は先頭の空白を落とす** (`run.c:421`)。SentencePiece のデコーダがそうするため (llama2.c PR #89)。
2. **`<0xNN>` は生バイトに直す** (`run.c:425`)。`sscanf(piece, "<0x%02hhX>", &byte_val)` で読むので、**厳密には後続の `>` を検査していない** し、16 進は 1 桁でも通る。実際の語彙は必ず `<0xNN>` の形なので問題にならないが、このリポジトリの正規表現 (`/\A<0x(\h{1,2})/`) は sscanf の挙動に合わせてある。

## 出力の規則

`safe_printf` (`run.c:431-443`) が **表示前にもう一段落とす**。

- 空の piece は捨てる
- **1 バイトだけの piece は、印字可能か空白でなければ捨てる**。生バイトのトークンは制御文字になりうるため
- 2 バイト以上の piece はそのまま通す (中身は検査しない)

`Llama2::Tokenizer#printable?` がこれにあたる。`isprint` は 0x20..0x7E、`isspace` は 0x09..0x0D と 0x20 (C ロケール)。

## 生成ループが印字するもの

`generate` (`run.c:747-773`) の順序が出力を決める。

1. `token` は最初のプロンプトトークン (プロンプトが空なら BOS)
2. logits を出し、`next` を選ぶ
3. **`next == 1` (BOS) なら打ち切る** — `run.c:763`。**EOS ではなく BOS が区切り** である
4. `decode(token, next)` を `safe_printf` に通す
5. `token = next`

**最初のトークン自身は印字されない。** 出力は 2 個目以降を、それぞれ 1 つ前のトークンとの組で復号したものになる。`Llama2::Tokenizer#render` がこの通りに書いてある (`each_cons(2)`)。

**PLAN-llama2.md には「EOS で停止する」と書いてあったが、llama2.c が見るのは BOS である。** EOS (id 2) は語彙にあるが、`generate` の打ち切り条件には出てこない。

## 検証

`./run stories260K.bin -z tok512.bin -t 0.0 -n 200` の出力は llama2.c の `test_all.py:37` に文字列として公開されている。このリポジトリは **その文字列を fixture に持ち、Ruby 実装の出力とバイト単位で一致することを要求する** (test/test_llama2_generate.rb)。stories110M についても、vendor の `run.c` をビルドした出力と 64 トークンで一致することを確認した。
