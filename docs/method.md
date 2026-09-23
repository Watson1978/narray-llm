# 比較の条件と手順

同じ重み・同じアルゴリズムで実装を並べ、tokens/sec を比べるときの共通の条件と手順。**結果そのものは docs/results/ の実装ごとのファイルにある** (現在は [gpt2-124m.md](results/gpt2-124m.md))。機械の側の性質は [machine.md](machine.md)、cumo の版ごとの所見は [cumo-history.md](cumo-history.md)。

## 比較条件

**重みとトークン列の fixture は実装ごとに違う。** 以下は GPT-2 124M の場合で、実装が増えたらその実装のファイル名に読み替える。

- 重みは同一ファイル `data/gpt2_124M.bin` (497,904,640 バイト) を両方が読む
- 生成は EOT 1 個からの貪欲法。**4 実装すべてが Ruby の生成トークン列と完全一致** (`python/fixtures/gpt2_124M_greedy.json` との照合、64 と 256 の両方)
- best-of-3、ウォームアップ 1 回、区間内に同期を挟まない。CuPy は計測の終わりで `cupy.cuda.Stream.null.synchronize()`
- プロセス起動・import・重み読み込みは計測に含めない
- dtype は fp32 のみ

### 版

README の表を測ったときの版は README の「計測環境」にある。下は [results/gpt2-124m.md](results/gpt2-124m.md) の初期の比較表を測ったときの版で、経緯として残してある。

| | 版 |
|---|---|
| Ruby | 4.0.6 / numo-narray-alt 0.11.2 / **cumo 0.7.0** |
| Python | 3.14.7 / numpy 2.5.2 (scipy-openblas 0.3.34) / cupy-cuda13x 14.2.0 / **torch 2.13.0+cu130** |
| | GPU の比較表を測り直した時点では numpy が 2.5.3 に上がっている (CuPy を入れ直した副作用)。CPU 表の値は 2.5.2 で測ったもの |
| CUDA | runtime 13.2 / driver 610.57.04 |
| GPU | RTX 5070 Ti Laptop (sm_120) |

**現在の開発環境は Ruby 4.0.7 / cumo 0.8.0 である。** 表の tokens/sec は測り直していない。0.8.0 は 0.7.0 と同一セッションでインターリーブして 4 条件 × 10 ラウンド測り、**どの条件でもペア比の範囲が 1 をまたいだ** ([results/gpt2-124m.md](results/gpt2-124m.md))。

**この先「cumo の版だけを変えた比較」は取れない。** Ruby 4.0.6 は rbenv から消えていて、過去の数字は全部その上で測ってある。0.7.0 対 0.8.0 のペアだけは、0.7.0 の拡張を 4.0.7 で入れ直してから測ったので版の差だけになっている (両方の `gem_make.out` で `arch=compute_120,code=sm_120` が一致することも確認した)。

cumo は #287〜#292 で「スカラーを 0 次元配列にせず数値のままカーネルへ渡す」最適化が入り、#296 で左オペランドがスカラーの場合 (`0.5 * a`) も同じ経路に乗った。#297 では添字が確保していないインデックス配列を解放しなくなった。いずれも 0.5.10 に入っている。GPU 列はそのたびに測り直しているが、**#297 では tokens/sec は動いていない** (後述)。0.5.11 (#307〜#334) はこのワークロードの通る経路に触れていない。**0.6.0 (#335〜#364) は触れていて、キャッシュ無しの経路が実際に速くなった。** 0.7.0 (#365〜#422) は転置ビューのコピーを速くしたが、それはモデル読み込みの経路で、生成には出ない。**いずれも [results/gpt2-124m.md](results/gpt2-124m.md) に測定がある。**

GPU 列は **released 0.7.0 の gem をそのまま入れて測った** (`gem install cumo -v 0.7.0`)。**この表は `gem install` から再現できる。** 0.5.10 から 0.5.11 までは同じ値で、0.6.0 でキャッシュ無しの 2 行が動き、0.7.0 では判別できなかった (いずれも [results/gpt2-124m.md](results/gpt2-124m.md))。0.5.10 は v0.5.9 の 40 コミット後で、PR 番号では #268〜#306 にあたる。このワークロードに効くのは #268〜#297 の側で、#298〜#306 はここが通らない経路の修正である (実際、4 条件の tokens/sec は開発中に使っていた `07f0af91` ビルドと中央値で +0.07〜+1.0% しか違わず、下のプロセス間の幅の内側にある)。**released 0.5.9 との差のほうは大きい。** #286 の時点では 64 トークン生成が 382.99 tok/s で、**同じ実装で測った 0.5.10 の 430.78** とは 1 割強の開きがある (内訳は docs/cumo-history.md)。ただしこの行のプロセス間の幅は 10.5% あるので、開きの大きさは幅を踏まえて読むこと。

**版が変わるたびに 4 条件を測り直している。** 0.5.11 は 0.5.10 と判別できず、**0.6.0 は KV キャッシュ無しの 2 行を実際に速くし** (256 トークンで +6.4%)、0.7.0 はまた判別できなかった。いずれも同一セッション内で 2 ビルドをインターリーブして測っている。

PyTorch の SDPA は実行時に選ばれたバックエンドを毎回出力している。CUDA では **prefill (T=64) も decode (T=1) も `efficient`** が選ばれた (`can_use_cudnn_attention` / `can_use_flash_attention` / `can_use_efficient_attention` を実際の形状で問い合わせ、有効かつ受理する最初のものを報告している)。CPU 側は `can_use_*` に相当する API が無いため判定していない。

**この「天井」は fp32 の天井である。** flash バックエンドは fp16 / bf16 専用なので、fp32 に縛った本比較では選ばれようがない。bf16 + flash まで解禁すればさらに上があるが、それをするとトークン列の完全一致という本比較の前提が壊れるため範囲外にしてある。結果のファイルに出てくる「天井の 8 割」もこの意味での天井を指す。**前提が壊れることは実際に測ってある** ([results/gpt2-124m.md](results/gpt2-124m.md) の 「fp16 (Cumo::HFloat) で測るとどうなるか」)。

### TF32 は無効

TF32 が有効だと速度も数値も変わり、トークン一致が壊れる。CuPy は `CUPY_TF32=0` (既定) で、実行時に cuBLAS の math mode が 0 (`CUBLAS_DEFAULT_MATH`) であることを毎回出力して確認している。

PyTorch はフラグが 2 つあり、どちらも Ampere 以降で既定が有効なので **両方を落とす**。`torch.backends.cuda.matmul.allow_tf32 = False` と `torch.backends.cudnn.allow_tf32 = False` で、実行時に両方の値を出力して確認している。

### CPU 側の BLAS

Numo の `dot` が BLAS を通るのは `Numo::Linalg` が定義されているときだけで、無ければ broadcast + `mulsum` にフォールバックする。**この計測では `numo-linalg-alt` を入れて BLAS 経路にしてある** (`lib/narray_llm/backend.rb` が `numo/linalg` を任意 require する)。入れないと同じ GEMM が 27 倍遅くなり、比較にならない。

両者ともリンク先は OpenBLAS だが **別のビルド** で、スレッド数を揃えても差が残る。`[256,768] x [768,2304]` の GEMM:

| | 1 スレッド | 24 スレッド |
|---|---|---|
| Numo (numo-linalg-alt 同梱 OpenBLAS 0.3.34) | 131.83 | 646.93 |
| NumPy (scipy-openblas 0.3.34) | 149.58 | 1276.84 |
| Numo、BLAS 無し (参考) | 23.88 | — |

1 スレッドではほぼ互角 (0.88 倍)、24 スレッドでは NumPy が 2.0 倍。同じ 0.3.34 でもビルドオプションが違うためで、**言語やバインディングの差ではない**。下表はスレッド数を揃えた行どうしで読む。

## 実装の対応関係

Ruby 側は途中で attention のヘッドをまとめる変更を入れたため、比較は 2 段構えになる。**構造が違うものを並べても意味がないので、対応する組み合わせで読む。**

| 構造 | Ruby | Python |
|---|---|---|
| ヘッドごとのループ | commit 470ac7a | `python/gpt2.py` 既定 |
| 全ヘッドをまとめる | 現行 HEAD | `IDIOMATIC=1` |

**バッチ経路 (`--batch`) はこの表の外にある。** batch > 1 では 3 実装とも必ず「全ヘッドをまとめる」形を通る (ヘッドごとのループにバッチ版を書いていない)。**`IDIOMATIC` が効くのは batch 1 だけである。**

**ここで 1 度取り違えた。** バッチの比較表を既定のまま測って **Cumo / CuPy = 9.47 倍** と出たが、**batch 1 の行だけがヘッドごとのループと比べた数字だった**。`IDIOMATIC=1` で揃えたら 3.06 倍になる。**同じ表の中で batch 1 と batch 8 が違うアルゴリズムを指していた** ので、行どうしを比べても気づけない形だった。**バッチを測るときは `IDIOMATIC=1` を付けること。**

**PyTorch (`python/gpt2_torch.py`) はこの対応表の外にある。** 構造の鏡写しは求めず、融合カーネルを含めた「エコシステムの実力値」を天井の参照点として置くためのもの。`F.scaled_dot_product_attention` を使い、LayerNorm や GELU も torch の実装をそのまま使う。したがって PyTorch の列は Numo / Cumo / NumPy / CuPy と同じ土俵の数字ではなく、**同じ重みと同じ手順で到達できる上限** として読む。

そのうえで、比較の前提になる条件は揃えてある。同じ `gpt2_124M.bin`、fp32 のみ、TF32 無効、prefill / decode の 2 相、`[maxT, C]` の KV キャッシュ、貪欲法、同一の計測プロトコル。全体を `torch.inference_mode()` で包んである (忘れると autograd の記録コストを払った数字になり、天井として不当に低く出る)。`torch.compile` は使っていない。初回コンパイルの扱いで計測の話が別物になるため、eager のみ。

まとめ方も鏡写しにしてある。全系列の attention は `[B*NH, T, hs]` の 3 次元バッチ行列積、decode は行列積を使わず要素積 + 縮約 (ヘッドごとだと `[1,hs] x [hs,t]` と小さすぎて GEMM 呼び出しに見合わないため)。Python 側で `einsum` を使えば 1 行になるが、縮約順序を自分で選ぶため「同じ演算を走らせる」前提が崩れるので使っていない。

## 再現方法

README の表はすべてここの手順で測った。版は README の「計測環境」にある。

### 準備

```
bundle config set --local with gpu
CUMO_NVCC_GENERATE_CODE=arch=compute_120,code=sm_120 bundle install

python3 -m venv python/.venv
python/.venv/bin/pip install -r python/requirements.txt
# PyTorch は同梱 CUDA が版で変わるので index を指定して別に入れる
python/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130

rake download
rake prepare                     # 変換した重み、参照値、Python 側が照合するトークン列
```

`CUMO_NVCC_GENERATE_CODE` は初回起動の JIT を避けるためで、sm_120 以外の GPU では値を読み替える。

### クロックを固定する

```
sudo nvidia-smi -lgc 3090
sudo nvidia-smi -lmc 14001       # 2 つは 1 回ずつ別に実行する
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader   # アイドルで SM 2790〜2805 MHz なら掛かっている
```

終わったら `sudo nvidia-smi -rgc` と `sudo nvidia-smi -rmc` で戻す。計測の前には、GPU を使うプロセスが無いこと (`nvidia-smi --query-compute-apps=pid,name --format=csv`) と、load average が低いことも確かめる。CPU が埋まっていてもカーネルの投入が遅れて wall に出る。

### 3 実装の表 (GPT-2、Llama 2、Mamba、Switch、Whisper)

```
bench/run.sh bench/three_impl.tsv tmp/bench
python3 bench/aggregate.py bench/three_impl.tsv tmp/bench
```

`bench/three_impl.tsv` の 1 行が 1 条件で、Cumo、CuPy、PyTorch のコマンドを並べてある。`bench/run.sh` はそれに対照 (Cumo をもう一度) を足した 4 系列を 1 ラウンドにまとめ、系列の順序をラウンドごとに回転させて直列に走らせる。ラウンドは 11 (`ROUNDS=10` で 0〜10) で、`bench/aggregate.py` が先頭の 1 本を位置で捨てて 10 ラウンドで集計する。どれか 1 本でも失敗するか数字が取れなければ、バッチごと止まる。

条件ごとの `--rounds` は、1 プロセスが 2 秒以上続けて回るように選んである。計測区間が短いとメモリクロックの段が上がる前に終わり、走行ごとに 9001 と 14001 のどちらかに落ちる (AGENTS.md)。Python 側は計測の前に生成したトークン列を fixture と照合し、ずれていれば数字を出さずに失敗する。

条件は 18 本のバッチで測り、幅が 8% を超えた条件だけ `--rounds` を上げて測り直した。`bench/three_impl.tsv` はその最終的なコマンドである (経緯は [cumo-history.md](cumo-history.md) の「3 実装の表を 0.10.0 で測り直した」)。

### int8 の表

3 実装の表と同じ 4 系列の手順で、モデルだけを `stories110M_q80` にした。2 条件をインターリーブして 10 ラウンド。結果と経緯は [results/llama2-int8.md](results/llama2-int8.md) の「3 実装を並べる」。

```
GPU=1 ruby script/llama2_generate.rb stories110M_q80 --length 64
GPU=1 python/.venv/bin/python python/bench_llama2.py --model stories110M_q80 --length 64 --cache 1
GPU=1 python/.venv/bin/python python/bench_llama2.py --impl torch --model stories110M_q80 --length 64 --cache 1
```

### ResNet-18 の表

1 つ目の表 (`shift` と `unfold`) は、Cumo の 2 綴り、CuPy の 2 綴り、対照 (Cumo `unfold`) の 5 条件をインターリーブして 11 ラウンド回し、先頭を捨てて 10 ラウンド。`--inner` は 1 条件が約 2 秒になる回数にする。

```
GPU=1 ruby script/resnet_classify.rb --spelling unfold --inner 120 --no-check
GPU=1 python/.venv/bin/python python/bench_resnet.py --spelling unfold --inner 120
```

2 つ目の表 (cuDNN) は 6 条件、13 ラウンドの先頭を捨てて 12 ラウンドで、上限の既定が 8 MiB で畳み込みが TF32 に載りうる版の cumo (0.9.0 以前) で測った。ワークスペースの上限は `CUMO_CUDNN_MAX_WORKSPACE_SIZE` (バイト数)、PyTorch の TF32 は `--no-tf32` で切る。`--inner` はこちらも 1 条件が約 2 秒になる回数にする。cumo 0.10.0 以降は上限の既定が 128 MiB で、TF32 は `CUMO_ALLOW_TF32=1` のときだけ使う。

```
GPU=1 ruby script/resnet_classify.rb --spelling cudnn --inner 250 --no-check
GPU=1 CUMO_CUDNN_MAX_WORKSPACE_SIZE=1073741824 ruby script/resnet_classify.rb --spelling cudnn --inner 250 --no-check
GPU=1 python/.venv/bin/python python/bench_resnet.py --impl torch --inner 250
GPU=1 python/.venv/bin/python python/bench_resnet.py --impl torch --inner 250 --no-tf32
```

結果と経緯は [results/resnet-18.md](results/resnet-18.md) の「3 実装を並べる」と「cuDNN のワークスペースが 8 MiB に制限されていた」。

### サンプリングの表

GPT-2 の長さ 900 で、貪欲法、`--top-k 50`、`--top-p 0.9`、両方、両方 (ソートを共有しない旧版)、対照 (貪欲法) の 6 条件をインターリーブして 11 ラウンド、先頭を捨てて 10 ラウンド。計測用の生成は EOT で止まらない。README の表に載せたのは旧版を除く 5 条件である。

```
GPU=1 ruby script/gpt2_generate.rb --length 900
GPU=1 ruby script/gpt2_generate.rb --length 900 --top-k 50 --top-p 0.9
```

結果と経緯は [results/gpt2-124m.md](results/gpt2-124m.md) の「サンプリング」。

### 学習の表

1 ステップの区間ごとに、Cumo と PyTorch の 3 区間 (forward、backward、update) と対照の 7 条件をインターリーブして 11 ラウンド、先頭を捨てて 10 ラウンド。1 回は 40 ステップで、先頭の 1 ステップを除いた平均を取る。README の表は、その s/step の逆数である。

```
GPU=1 ruby script/gpt2_train.rb --steps 40 --stop-after backward
GPU=1 python/.venv/bin/python python/bench_gpt2_train.py --impl torch --steps 40 --stop-after backward
```

勾配と損失の一致 (関門 B と C) は `rake test` が確かめる。結果と経緯は [results/gpt2-124m.md](results/gpt2-124m.md) の「学習」。

### カーネル数

decode 1 トークンあたりのカーネル数は nsys で数える。`--no-bench` を付けて生成 1 回だけにすると、差の割り算がちょうど 32 になる。**2 本とも 40 トークン以上にすること** (短いほうが冷えていると時間が過小に出る。理由は AGENTS.md の計測の作法 8 番)。

```
GPU=1 nsys profile --trace=cuda --cpuctxsw=none --sample=none -f true -o dec72 \
  ruby script/gpt2_generate.rb --length 72 --no-bench
GPU=1 nsys profile --trace=cuda --cpuctxsw=none --sample=none -f true -o dec40 \
  ruby script/gpt2_generate.rb --length 40 --no-bench
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv dec72.nsys-rep
nsys stats --force-export=true --report cuda_api_sum      --format csv dec72.nsys-rep
```

カーネルごとに `(dec72 の Instances - dec40 の Instances) / 32` が 1 トークンあたりの本数。オプションは `--sample=none` (`--sampling=none` は存在しない)、`--force-export=true` を付けないと古い .sqlite が黙って使われる。CSV の列は 0 始まりで 2=Instances、3=Avg、4=Med、6=Max、8=Name。**時間は Avg × 本数を原則とし、Max / Med が 100 を超える行だけ Med × 本数に置き換える** (理由は上記)。

1 演算あたりの固定費は `bench_gpt2_sweep.py` が毎回出力する。Ruby 側は次で測る。

**GPU を実負荷で温めてから測ること。** `a + b` を 100 回回すだけでは GPU のクロックが上がらず、冷えた状態 (960 MHz) では 2.99 us、温まった状態 (2767 MHz) では 2.06 us と 1.45 倍ずれる。AGENTS.md の「単発測定は GPU クロックの立ち上がりでぶれる」がそのまま出る箇所。

```
GPU=1 ruby -Ilib -e 'require "narray_llm"
  def sync; XM::CUDA::Runtime.cudaDeviceSynchronize; end
  # 先に実負荷でクロックを上げる
  m = NArrayLLM::GPT2::Model.load("data/gpt2_124M.bin")
  tk = NArrayLLM::GPT2::Tokenizer.load("data/gpt2_tokenizer.bin")
  g = NArrayLLM::Generator.new(m, tokenizer: tk)
  3.times { g.generate([tk.eot_token], max_new_tokens: 64) }; sync
  a = XM::SFloat.ones(256); b = XM::SFloat.ones(256)
  100.times { a + b }; sync
  t = Array.new(5) { sync; s = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    1000.times { a + b }; sync
    Process.clock_gettime(Process::CLOCK_MONOTONIC) - s }.min / 1000
  puts format("%.2f us", t * 1e6)'
```
