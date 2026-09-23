# Mamba のトークナイザ形式

[kroggen/mamba.c](https://github.com/kroggen/mamba.c) の `tokenizer.py` が書く表。出典はそのスクリプトと `mamba.c` の `build_tokenizer` (`mamba.c:612`)、`decode` (`mamba.c:570`)、`safe_printf` (`mamba.c:583`)。

## 中身は GPT-NeoX

`EleutherAI/gpt-neox-20b` のトークナイザを `transformers` で読み、**`tokenizer.decode([i])` の結果を UTF-8 バイトとして 50277 個書き出したもの** である。

**GPT-2 の byte-level BPE のマッピングは書き出しの時点で解決済み** なので、読む側はバイト列をそのまま使う。llama2.c の SentencePiece と違い、**`<0x0A>` のような生バイトを綴った語彙は出てこない**。

## 作り方

```
cd vendor/mamba.c && ../../python/.venv/bin/python tokenizer.py
mv tokenizer.bin ../../data/mamba_tokenizer.bin
```

`transformers` が要る。表そのものは 514,694 バイト。

## 並び

| バイト | 内容 |
|---|---|
| 0〜3 | マジック `0x4d62546b` (`"MbTk"`)、uint32 |
| 4〜7 | バージョン `1`、uint32 |
| 8〜11 | トークン数、uint32 |
| 12〜15 | 最大トークン長、uint32 |
| 16〜 | `(長さ uint32, バイト列)` の繰り返し |

**スコアが無い。** llama2.c の表は 1 語彙につき `(score float32, 長さ int32, バイト列)` だが、こちらは長さとバイト列だけである。エンコーダが要らないので、マージ順を決めるスコアも要らない。

**ヘッダがトークン数を持っている。** llama2.c の表は持っておらず、モデルの `vocab_size` から渡す必要があった。こちらは自己記述的である。

mamba-130m では **トークン数 50277、最大トークン長 512**。

## デコード

`mamba.c:570` の `decode` は 2 つのことをする。

1. **直前が区切りトークンなら、先頭の空白を 1 つ落とす** (`prev_token == EOS && piece[0] == ' '`)
2. `<0x..>` の形なら生バイトに直す — **GPT-NeoX では発火しない** が、llama2.c から引き継がれている

**区切りは `<|endoftext|>` で、id は 0。**`mamba.c:506` は `BOS` と `EOS` を両方 0 に定義している。id 1 は `<|padding|>`。

`safe_printf` (`mamba.c:583`) は **1 バイトの piece が印字可能でも空白でもなければ落とす**。複数バイトの piece は素通しする。

## C 側の制限

**`mamba.c` は piece を NUL 終端で持つ** (`t->vocab[i][len] = '\0'`)。語彙の途中に NUL バイトがあれば C 側では切れるが、**このリポジトリは長さで保持する** ので切れない。GPT-NeoX の語彙に NUL は含まれないので、いまのところ差は出ない。

## 検証

**表は上流と突き合わせた。**`transformers` で `EleutherAI/gpt-neox-20b` を読み、`decode([i])` の結果と比較して一致することを確認した (id 0, 1, 187, 209, 15496, 50000, 50275, 50276)。**ファイルを自分自身と比べるのではなく、出どころと比べている。**
