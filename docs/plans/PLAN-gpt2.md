# GPT-2 124M 実装プラン

GPT-2 124M の推論を Numo::NArray / Cumo だけで実装する。llm.c の重み・参照値をそのまま使い、数値一致を各段階の関門とする。
**第 0〜3 段階は完了している。** 次の実装 (Llama 2) のプランは [PLAN-llama2.md](PLAN-llama2.md) にあり、下の「全体の約束事」をそのまま継ぐ。

## 全体の約束事

- バックエンドは環境変数で切り替える。既定は Numo、`GPU=1` で Cumo。

  ```ruby
  if ENV['GPU'].to_s =~ /\A(1|on|true)\z/i
    require 'cumo/narray'
    XM = Cumo
  else
    require 'numo/narray'
    XM = Numo
  end
  ```

- dtype は SFloat (fp32) のみ。DFloat は使わない。
- Bit 配列・マスク代入・fancy index は原則使わない。使う場合は理由をコメントに残す (詳細は AGENTS.md の「地雷」参照)。
- 依存 gem は最小にする。第 0〜2 段階は numo-narray / cumo のみで完結させる。BPE エンコーダは実装しない (入力はトークン化済みの列を使う)。
- 数値検証は二段構え:
  1. Numo の出力 vs llm.c の参照値 (実装の正しさ)
  2. Cumo の出力 vs Numo の出力 (バックエンド間の一致)
- 各段階の受け入れ条件は `rake test` で機械検証できる形にする。**前の段階のテストが通るまで次の段階のコードを書かない。**
- 計測は必ず best-of-N (N>=3)。単発測定の数字を信用しない。Cumo の計測は反復ごとに `cudaDeviceSynchronize` を挟む区間計測と、同期なしの総時間計測を区別する。

## 第 0 段階: データ取得と重みの読み込み

想定: 半日

### 作るもの

- `script/download_gpt2.rb` — llm.c の starter pack から以下を取得する:
  - `gpt2_124M.bin` (fp32 重み)
  - `gpt2_124M_debug_state.bin` (参照入力・参照 logits・参照 loss)
  - トークナイザの bin (語彙 → バイト列のテーブル)
- `lib/narray_llm/models/gpt2/checkpoint.rb` — `.bin` のローダ:
  - 先頭の int32 x 256 ヘッダからマジックナンバー・バージョン・モデル設定 (maxT, V, L, NH, C) を読む
  - パラメータ本体を `from_binary` + `reshape` で名前付きテンソル群に割り付ける
  - **注意**: パラメータの並び順・各テンソルの shape は llm.c の `gpt2_build_from_checkpoint` を読んで確定させる。推測で書かない。
  - **注意**: debug_state のフォーマット (何がどの順で入っているか) も llm.c の `test_gpt2.c` を読んで確定させる。これはこの段階の作業に含む。

### 受け入れ条件 (test/test_gpt2_checkpoint.rb)

- [ ] ヘッダから読んだ設定が GPT-2 124M の公称値と一致する (L=12, NH=12, C=768, V=50257, maxT=1024)
- [ ] 全パラメータの要素数の合計が checkpoint のデータサイズと一致する
- [ ] 各テンソルの shape 一覧が llm.c のコメント/コードと一致する (一覧を fixture としてテストに書き込む)
- [ ] wte など数本のテンソルについて、先頭数要素の値が参照と一致する

## 第一段階: フォワード 1 回

想定: 1〜2 日

### 作るもの

- `lib/narray_llm/models/gpt2/model.rb` — フォワードパス:
  - 埋め込み: **`wte[ids, true]` の gather で実装する** (当初ここには「one-hot 行列 x wte の GEMM で実装する。gather は同期が入るため」と書いていたが、**同期するのは NArray を添字にしたときだけ** で、Ruby の Array なら同期しない。`ids` は Ruby の Array のまま届くので gather が使える。キャッシュ無しで 1.22〜1.31 倍。[docs/results/gpt2-124m.md](../results/gpt2-124m.md))
  - ブロック: pre-LayerNorm → QKV GEMM → ヘッド分割 → attention → 射影 → 残差 → LayerNorm → MLP (GELU tanh 近似) → 残差
  - ヘッド分割の列スライスは連続化してから `dot` に渡す
  - 因果マスクは Bit を使わず算術で作る (`clip` + `ceil`)
  - 最後に LayerNorm → wte との共有重みで logits (unembedding も GEMM)
- `script/gpt2_forward.rb` — debug_state の入力 (B=4, T=64) を流して logits を参照値と比較するランナー
- 層ごとの中間活性を参照値と比較し、**どの層から乖離が始まるかを表示する仕組み** を最初から入れる (第一段階のデバッグの主武器になる)
- 層ごとの時間内訳 (embed / gemm / softmax / layernorm / gelu) を出す

### 受け入れ条件 (test/test_gpt2_forward.rb)

- [ ] Numo の logits と参照 logits の最大絶対誤差が閾値内 (閾値は llm.c の test_gpt2 が使う許容値に合わせる。fp32 で 1e-2 オーダー)
- [ ] Cumo の logits と Numo の logits の乖離が、上の閾値より一桁小さい
- [ ] 参照 loss との一致 (debug_state に loss が入っている場合)
- [ ] B=1, T=1 の縮退ケースでも例外なく通る

## 第二段階: 生成

想定: 1 日

### 作るもの

- `lib/narray_llm/models/gpt2/tokenizer.rb` — **デコードのみ**。トークナイザ bin から id → バイト列の表を読む。エンコードは実装しない。
- `lib/narray_llm/generator.rb` — 貪欲法 (argmax) のみ。温度・top-k・サンプリングは第三段階の後まで入れない。
  - argmax は `max_index` を使う。**平坦化インデックスが返るのでオフセット補正を忘れない** (AGENTS.md 参照)
- `script/gpt2_generate.rb` — 同梱の固定プロンプト (トークン化済み) から N トークン生成して表示するランナー
- ベンチ指標: tokens/sec と 1 トークンあたりの時間内訳

### 受け入れ条件 (test/test_gpt2_generate.rb)

- [ ] 同じプロンプトから Numo と Cumo が **同一のトークン列** を生成する (argmax は決定的。割れたら logits の乖離が閾値を超えている証拠なので、テスト失敗として扱いつつ、どの位置で割れたかを表示する)
- [ ] 生成長 1 と生成長 N で先頭トークンが一致する (状態の持ち回りにバグがない)
- [ ] EOT トークンで停止する

## 第三段階: KV キャッシュ

想定: 1〜2 日

### 作るもの

- K, V を層ごとに `[maxT, C]` で事前確保し、ステップごとに 1 行ずつ書き足す
  - **ここで初めて非連続ビューへの書き込み・伸びるスライスの読み出しが主役になる。** cumo #245 以降の修正が効く経路そのもの
- 生成ループを「毎回全系列を再計算」から「新トークン 1 個 + キャッシュ参照」へ
- VRAM の見積もりを起動時に表示し、上限を超える生成長は拒否する (K+V で L x maxT x C x 2 x 4 bytes ≒ 75 MB + attention の一時配列。8GB 級でも動くが、B を増やす場合は要注意)

### 受け入れ条件 (test/test_gpt2_kv_cache.rb)

- [ ] KV キャッシュ有無で生成トークン列が **完全一致** する
- [ ] tokens/sec がキャッシュ無し比で改善し、系列長が伸びるほど差が開く
- [ ] maxT を超える生成要求を明示的なエラーで拒否する

### 最終成果物

- numo / cumo x キャッシュ有無の 4 点比較表 (tokens/sec、時間内訳つき)
- README にモデル取得手順・実行例・比較表を載せる

## 第四段階以降の候補 (このプランの範囲外)

- ~~gather 版埋め込みとの比較 (one-hot GEMM とどちらが速いか)~~ — **済み。** gather が速く、実装はそちらに移した
- 温度・top-k サンプリング
- ~~バッチ生成~~ — **済み。**[PLAN-batch.md](PLAN-batch.md) に進め方と結果がある
- GPT-2 medium/large への対応 (ローダは設定を読むので原理的には動くはず)
