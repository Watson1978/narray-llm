# Llama 2 実装プラン

Llama 2 の推論を Numo::NArray / Cumo だけで実装する。**目的は cumo の検証** で、GPT-2 が踏んでいない経路を叩くことに価値がある。参照実装は [llama2.c](https://github.com/karpathy/llama2.c) で、GPT-2 における llm.c と同じ役割を果たす。

進め方は [PLAN-gpt2.md](PLAN-gpt2.md) の「全体の約束事」をそのまま継ぐ。**前の段階の受け入れテストが通るまで次の段階のコードを書かない。**

## なぜ Llama 2 か (cumo の検証として)

| 新しく叩く経路 | GPT-2 では |
|---|---|
| **RoPE** — `[..., hs/2, 2]` に reshape してペアを回転。**ストライド 2 の非連続ビューとその再合成** | 踏まない。AGENTS.md が「非連続ビューは cumo が性能崖を踏みやすい (#245〜#252)」と書いている経路 |
| **RMSNorm** — 中心化しない縮約。fp16 では `x²` がそのまま乗る | layernorm の `(x-mean)²` は測ってある (余裕 5 倍)。中心化しない場合との比較になる |
| **SwiGLU** — FFN が 2 行列から 3 行列に。decode の gemv が 1 本増える | 2 行列 |
| **偶数の語彙** (32000) | 50257 は奇数。docs の「奇数の列は gemv には効かない」を別の数で検証できる |
| **GQA** | 無い。**stories260K が既に GQA なので第一段階で踏む** (下記) |

## コードの置き場 (決定済み)

`NArrayGPT2` / `lib/narray_gpt2/` を **`NArrayLLM` / `lib/narray_llm/` に改名済み**。共有するものと実装固有のものを分けてある。

```
lib/narray_llm.rb
lib/narray_llm/
  backend.rb binary_io.rb compare.rb profiler.rb    共有
  ops.rb kv_cache.rb generator.rb                   共有
  models/
    gpt2.rb                                         GPT-2 の require をまとめる
    gpt2/checkpoint.rb model.rb tokenizer.rb debug_state.rb
```

実装固有のものは `NArrayLLM::GPT2` に入れてある (`NArrayLLM::GPT2::Model` など)。**Llama 2 は `models/llama2.rb` と `models/llama2/` を足し、`NArrayLLM::Llama2` に入れる。**

**`Ops` に何を共有させるかは第一段階で決める。**`rmsnorm` / `silu` / `rope` は今のところ Llama でしか使わないが、共有の `Ops` に置くか `Llama2::Ops` に分けるかは、GPT-2 側と並べて測るときに都合のよいほうを選ぶ。

## 第 0 段階: 重みの取得とローダ

想定: 半日

### 作るもの

- `script/download_llama2.rb` — チェックポイントとトークナイザを取得する
  - `stories260K` (形式の検証用。数百 KB で速い)
  - `stories110M` (計測用。GPT-2 124M と同規模)
  - `tokenizer.bin` (llama2.c 同梱。260K 用は語彙 512 の別ファイル)
- チェックポイントのローダ
  - ヘッダは `Config` の int32 x 7 (dim / hidden_dim / n_layers / n_heads / n_kv_heads / vocab_size / seq_len)
  - **`vocab_size` が負なら分類器の重みを共有しない** という符号ハックがある
  - 重みの並びは `token_embedding_table → rms_att_weight → wq → wk → wv → wo → rms_ffn_weight → w1 → w2 → w3 → rms_final_weight → (freq_cis は読み飛ばし) → wcls`
  - **注意: 並びと shape は `run.c` の `memory_map_weights` を読んで確定させる。推測で書かない。** GPT-2 のときに llm.c の `gpt2_build_from_checkpoint` を読んだのと同じ手順
- `docs/checkpoint-format-gpt2.md` の隣に llama2.c 形式の仕様を書く (出典つき)

### 受け入れ条件

**第 0 段階は完了している** (test/test_llama2_checkpoint.rb、13 件)。

- [x] ヘッダから読んだ設定が stories260K / stories110M の公称値と一致する
- [x] 全パラメータの要素数の合計がファイルサイズと一致する (ヘッダ 28 バイトを引いた残り)。**両モデルとも 1 バイトの差もなく一致した**
- [x] 共有分類器 (`vocab_size > 0`) と非共有 (`< 0`) の両方を正しく扱う。**配布されている 2 つはどちらも共有なので、非共有は仕様から組み立てた合成ファイルで押さえた**
- [x] 数本のテンソルについて先頭数要素が `run.c` の読み出しと一致する。**`memory_map_weights` のポインタ演算を手で展開したバイトオフセットを fixture に持ち、`unpack` で直読みした値と突き合わせている**

### 第 0 段階で分かったこと

**stories260K は GQA だった** (`n_heads` 8 / `n_kv_heads` 4、`kv_mul` 2)。形式検証用に選んだ最小のモデルが multiquery なので、**第一段階で GQA を実装することになる**。110M のほうは `n_heads == n_kv_heads` で素の MHA。

**どちらのモデルも分類器を共有している。** 負の `vocab_size` の経路は実ファイルでは踏めない。

**`run.c` の構造体のコメントは shape の向きを信用できない。**`wq` を `(layer, dim, n_heads * head_size)` と書いているが実際は出力が先である。`matmul` の引数と `export.py` が書き出す `nn.Linear` の `weight` の両方から決めた (docs/checkpoint-format-llama2.md)。

**`hidden_dim` は 172 で 32 の倍数ではない** (stories260K)。`dim` から計算で出そうとしないこと。

**freq_cis は読み飛ばすときも読むこと。**`seek` で飛ばすと、その領域で切り詰められたファイルが EOF 検査を素通りする。テストを書いて初めて気づいた。

## 第一段階: フォワード 1 回

想定: 2〜3 日 (RoPE と GQA のぶん GPT-2 より重い)

### 作るもの

- 参照値の生成器 — **llama2.c 側に無いので自分で作る。**`run.c` か llama2.c の `model.py` に手を入れて、**層ごとの中間活性と最終 logits を GPT-2 の debug_state と同じ形で吐く**。第一段階のデバッグの主武器になるので最初に作る
- `Ops` に足すもの:
  - `rmsnorm(x, weight, eps:)` — `x / sqrt(mean(x²) + eps) * weight`。**中心化しない**
  - `silu(x)` — `x * sigmoid(x)`。`exp` の引数は事前に clip する (AGENTS.md)
  - `rope(x, pos)` — `[..., hs/2, 2]` の reshape とペアの回転。**Bit も fancy index も使わない**。sin/cos の表は事前計算して持つ
- モデル本体: RMSNorm → QKV → RoPE → attention (因果マスクは GPT-2 と同じ算術) → 射影 → 残差 → RMSNorm → SwiGLU (`w2(silu(w1(x)) * w3(x))`) → 残差
- 層ごとの乖離表示は GPT-2 の `compare.rb` をそのまま使う

`run.c` を読んで確定させた細部 (第 0 段階のついでに確認したもの):

- **RoPE は q には全体に、k には `i < kv_dim` の範囲にだけ掛かる** (`run.c:274` の `rotn`)。GQA では k が短いので、q と同じ幅で回すと範囲外まで回してしまう
- **周波数は `head_dim = i % head_size` で決まる** (`run.c:267-269`)。ヘッドをまたいで通し番号にしないこと
- **クエリヘッド `h` が見る K/V ヘッドは `h / kv_mul`** (`run.c:295`、`:310`)。整数除算なので、隣り合う `kv_mul` 本の q が同じ K/V を共有する
- **スケールは `1/sqrt(head_size)`** で GPT-2 と同じ (`run.c:301`)

### 受け入れ条件

**第一段階は完了している** (test/test_llama2_forward.rb、16 件)。

- [x] Numo の logits と参照 logits の最大絶対誤差が閾値内 (閾値の根拠をコメントに書く)。**実測 1.0e-05 (260K) と 2.3e-05 (110M)、閾値 1e-3**
- [x] Cumo の logits と Numo の logits の乖離が、上の閾値より一桁小さい。**相互差 6.7e-06**
- [x] **RoPE 単体のテスト**: 既知の位置・既知の入力で参照と一致する。回転の向き (`x0·cos − x1·sin`) を取り違えても全体の誤差は小さく出るので、単体で押さえる
- [x] **RMSNorm 単体のテスト**: 中心化しないこと (平均が 0 でない行で layernorm と違う答えになること) を明示的に検査する
- [x] **GQA 単体のテスト**: `kv_mul` 本のクエリヘッドが同じ K/V ヘッドを見ること
- [x] stories260K と stories110M の両方で通る

**閾値に頼らない検査も入れてある。** 参照は `run.c` が自分の argmax を食わせて作ってあるので、位置 p の logits が選ぶトークンは位置 p+1 の入力そのものになる。**両モデルとも貪欲法のトークン列が完全一致する。** 埋め込みは行のコピーなので、one-hot GEMM が **完全一致 (max|d| = 0)** することも要求している。

### 参照値の生成器

`script/llama2_dump.c` が `run.c` を丸ごと include し、`forward` の制御フローだけを写して各段を書き出す。`rmsnorm` / `softmax` / `matmul` は `run.c` のものをそのまま使うので、**演算は参照側のもの** である。

**写した制御フローが唯一のリスクなので、そこは直接検査している。** 同じバイナリが 2 つ目の Transformer を作って `forward` を走らせ、毎ステップ logits を突き合わせる。**max|d| = 0 (完全一致)** でなければ `script/llama2_dump.rb` が止まる。加えて vendor の `run.c` をそのままビルドした出力が、`test_all.py` の公開している 200 トークンの既知出力と **バイト単位で一致する** ことも確認した。

```
ruby script/download_llama2.rb   # 重み・トークナイザ・run.c
ruby script/llama2_dump.rb       # data/<model>_debug_state.bin を作る
ruby script/llama2_forward.rb    # 段ごとの乖離を表示 (--layers で層ごと)
```

### 第一段階で分かったこと

**段ごとの乖離は全部 fp32 の丸め水準で、どこも他を打ち消していない** (stories260K、全 8 位置 × 全 5 層の最大)。`embed` が 0、それ以外が 1e-6〜1e-5、最後の `logits` が 1.0e-05。

**`Ops` には共有として置いた。**`rmsnorm` / `silu` / `rope` / `repeat_kv_heads` は今のところ Llama でしか使わないが、`layernorm` / `gelu` / `softmax_rows` と並べて測れるほうが都合がよいので `Llama2::Ops` に分けていない。

**K/V ヘッドの複製は broadcast で書いた。**`zeros(t, kv, kv_mul, hs) + x.reshape(t, kv, 1, hs)` で、添字配列がデバイスに渡らない。repeat との速さの比較は第四段階以降。

## 第二段階: 生成

想定: 1 日

### 作るもの

- トークナイザのデコード — `tokenizer.bin` の形式を読む。**エンコードは実装しない** (GPT-2 と同じ方針)。llama2.c の SentencePiece 系は GPT-2 の byte-level BPE と別形式なので、`docs/tokenizer-format-gpt2.md` の隣に仕様を書く ([docs/tokenizer-format-llama2.md](../tokenizer-format-llama2.md))
- 貪欲法の生成 — `Generator` は GPT-2 のものを共有できるはず。できなければ何が違うのかをコメントに残す

### 受け入れ条件

**第二段階は完了している** (test/test_llama2_generate.rb、11 件)。

- [x] **`run.c` を温度 0 で走らせた出力と、生成トークン列が完全一致する** (これがこの実装の関門。GPT-2 の fixture と同じ役割)。**stories260K の 200 トークンが `test_all.py` の公開文字列とバイト一致** し、stories110M も vendor の `run.c` の出力と 64 トークンで一致した
- [x] Numo と Cumo が同一のトークン列を出す
- [x] ~~EOS~~ **BOS** で停止する (下記)

### 第二段階で分かったこと

**打ち切り条件は EOS ではなく BOS だった。**`run.c:763` が `if (next == 1) { break; }` で、llama2.c では **BOS が系列の区切り** である。EOS (id 2) は語彙にあるが `generate` の条件には出てこない。stories260K は 200 ステップの間 BOS を出さないので、停止のテストはスタブのモデルで駆動している。

**`Generator` はそのまま共有できた。** 変更は 1 箇所だけで、`cache: true` を渡されたのに `new_cache` を持たないモデルだったときに、生成器の内部で落ちる代わりに明示的なエラーを出すようにした (KV キャッシュは第三段階)。

**トークナイザに語彙数が入っていない。**`run.c:387` のコメントが `i should have written the vocab_size into the tokenizer file... sigh` と言っているとおりで、**モデルの `Config` から渡す**。`Llama2::Tokenizer.load` が `vocab_size:` を要求するのはこのため。

**復号は 1 トークンでは閉じない。** BOS の直後だけ先頭の空白を落とす規則があるので、直前のトークンが要る。加えて `safe_printf` が **1 バイトだけの piece を印字可能か空白でなければ捨てる**。この 2 つを外すと出力がバイト一致しない。詳細は [docs/tokenizer-format-llama2.md](../tokenizer-format-llama2.md)。

```
ruby script/llama2_generate.rb                  # stories260K を 200 トークン
ruby script/llama2_generate.rb stories110M --length 64
```

## 第三段階: KV キャッシュ

想定: 1 日

### 作るもの

- GPT-2 の `KVCache` を共有する。**Llama では K に RoPE を適用してから積む** ので、その順序を間違えないこと (キャッシュに入れる前か後かで結果が変わる)

### 受け入れ条件

**第三段階は完了している** (test/test_llama2_kv_cache.rb、11 件)。

- [x] キャッシュ有無で生成トークン列が完全一致する。**両モデルで一致し、キャッシュ有りの出力も `run.c` の公開文字列とバイト一致する**
- [x] tokens/sec がキャッシュ無し比で改善する。**110M / GPU で 64 トークン 1.39 倍、200 トークン 2.16 倍** ([docs/results/llama2-110m.md](../results/llama2-110m.md) の 4 条件表、1 プロセス 1 条件で 10 ペア)。CPU では 64 トークン 2.7 倍、256 トークン 6.2 倍
- [x] `seq_len` を超える生成要求を明示的なエラーで拒否する。**`Generator` / `Model#decode` / `KVCache#append` の 3 箇所**

### 第三段階で分かったこと

**`KVCache` はそのまま共有できた。**`channels` で抽象化されているので、`kv_dim` を渡すだけでよい。**GQA のぶんキャッシュが狭くなる** のが GPT-2 との違いで、stories260K は `kv_dim` 32 と `dim` 64 の半分、つまり同じ `dim` の MHA に比べて `kv_mul` 分の 1 になる。

**K を RoPE の後に積むことは、参照値と直接突き合わせて確かめた。**`run.c:259` で `s->k` はキャッシュの行そのものを指し、`:279` の RoPE がその場で回すので、**積まれるのは回した後の鍵** である。参照の dump は `k` (RoPE 後) と `k_pre_rope` の両方を持っているので、prefill 後のキャッシュが前者と一致し、**後者とは一致しないこと** を両方テストしている (片方だけだと偶然通る余地が残る)。

**decode 側の attention は GQA で別経路になる。**`q` は `dim` 幅、キャッシュは `kv_dim` 幅なので、GPT-2 の `decode_attention` の形 (`keys * q` のブロードキャスト) がそのまま使えない。`Ops.decode_attention` は `num_kv_heads` が `num_heads` と等しいときは **従来の経路をそのまま通り**、違うときだけ `decode_attention_grouped` に入る。**GPT-2 側のカーネル数を動かさないため** にこの形にした。

**キャッシュを毎ステップ `kv_dim → dim` に広げてはいない。**`q` を `[1, kv, kv_mul, hs]` に割ってキャッシュ側を `[t, kv, 1, hs]` でブロードキャストする。層ごと・ステップごとに `[t, dim]` のコピーを作らずに済む。

```
ruby script/llama2_generate.rb stories110M --length 64
CACHE=0 ruby script/llama2_generate.rb stories110M --length 64   # 比較用
```

## 第四段階: 計測

想定: 1 日

### 作るもの

- `docs/results/llama2-110m.md` — GPT-2 と同じ 4 条件 (キャッシュ有無 x 生成長) の表
- Python 側の対応実装 (CuPy / PyTorch) — **GPT-2 のときと同じく「同じ形に書いた」ものにする**
- 1 トークンあたりのカーネル数 (nsys)

### 受け入れ条件

**第四段階は完了している** (結果は [docs/results/llama2-110m.md](../results/llama2-110m.md))。

- [x] 計測前にクロックの状態を確認している (AGENTS.md の計測の作法 14 番)。**`-lgc 3090` / `-lmc 14001` を掛けたうえで、実負荷を掛けながらサンプリングして段を確認した**
- [x] 4 条件とも 10 ペアのインターリーブで測ってある。**3 実装 × 4 条件 × 10 ラウンド = 120 プロセス**
- [x] **GPT-2 との差がどこから来るかを説明できる** (下記)

### 第四段階で分かったこと

**GPT-2 との差は、ほぼ丸ごと RoPE の書き方だった。** Cumo で同一セッションに並べると GPT-2 が 1.33〜1.37 倍速かったが、**`Ops.rope` を恒等写像に差し替えると順位がひっくり返って Llama 2 が 1.18〜1.19 倍速くなる** (どちらも 10/10、ペア比の範囲が 1 をまたがない)。

**そこで `reshape` を `reshape!` に置き換えたら、差が半分になった。**`reshape` は連続配列でもコピーするので、`Ops.rope` は 15 本のうち 6 本を形の変更に使っていた。書き直して 891 → 747 本、tokens/sec が 4 条件で +3.3〜19.2%、**GPT-2 との比が 1.33〜1.37 から 1.166 / 1.165 に縮んだ**。CuPy と PyTorch が 2% 以内で再現しているので、動いたのは Cumo だけである。

- **RoPE は 1 トークンあたり 360 本のカーネルを起動する** (891 本のうち 40%)。24 回の呼び出しで 360 本なので **1 回 15 本**
- 消すと wall も 1.59〜1.63 倍動くので、起動数だけの話ではない
- gemv は 49 → 85 本 (+36) だが、RoPE の 360 本に比べれば小さい
- 語彙が 50257 → 32000 になって読む重みは 494.6 → 438.1 MB (0.886 倍) と **減る**。逆向きに効くが、やはり小さい

**PLAN の「SwiGLU の gemv が 1 本増える」は過小だった。** 層あたり 4 → 7 本で **+3** である (QKV が 1 本から 3 本に割れて +2、FFN が +1)。

**層あたりの行列重みは GPT-2 と 1 要素も違わない** (`12C²` と `4D² + 3DH` がどちらも 7,077,888)。読む量の差は語彙だけから来る。

**`stories110M` は 243 トークン目で BOS を出して停止する** ので、生成長は 64 と 200 にした (256 だと条件ごとの仕事量が揃わない)。

**GQA のとき PyTorch の SDPA が `math` に落ちる。** stories260K (GQA) では `math`、stories110M (MHA) では `efficient` が選ばれた。fp32 かつ `enable_gqa` の組み合わせで実装が無いためと思われるが、**確かめていない**。

**次にやるなら RoPE の書き方である。** 今の実装は 2 つの非連続ビュー、4 つの積、2 つの和差、`concatenate` での再合成という素直な形をしている。融合する、ストライド代入を使う、cos/sin を `[t, head_size]` に広げて積 1 回で済ませる、といった余地が残っている。**どれが効くかは測っていない。**

## 第五段階以降の候補

- **より大きな GQA** — TinyLlama 1.1B を `export.py` で llama2.c 形式に変換して読む。**GQA 自体は第一段階の stories260K で踏む** ので、ここで新しいのは `kv_mul` が大きい場合と、K/V ヘッドを広げる経路の速さ (broadcast か repeat か、cumo でどちらが速いか) のほう
- fp16 — RMSNorm が中心化しないぶん、layernorm より早く溢れるかを測る
- MoE — エキスパート選択の gather は cumo で同期する。**同期なしで書けるか** が本題 (添字が Ruby の Array なら同期しない。NArray を添字にしたときだけ同期する)
