# narray-llm

6 モデル (+ Llama 2 の int8) の推論と GPT-2 124M の学習を [Numo::NArray](https://github.com/ruby-numo/numo-narray) / [Cumo](https://github.com/sonots/cumo) だけで実装するプロジェクト。深層学習フレームワークは使わない。

重みと参照値は [llm.c](https://github.com/karpathy/llm.c) と [llama2.c](https://github.com/karpathy/llama2.c) のものをそのまま使い、各段階の関門を参照実装とのトークン列の完全一致に置く。学習だけは完全一致にできないので、そこは llm.c 自身の許容誤差を借りている (下記)。同じコードが `XM` 定数の差し替えだけで CPU (Numo) と GPU (Cumo) の両方で動く。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.png">
  <img alt="narray-llm のアーキテクチャ: script/ から Generator と 6 つのモデル、Ops、XM と XF を経て Numo と Cumo に届くまで" src="docs/architecture-light.png">
</picture>

## In English

narray-llm implements six models in Ruby: GPT-2 124M, Llama 2 110M (also with int8 weights), Mamba 130M, Switch Transformer base-8, Whisper tiny and ResNet-18. It also trains GPT-2 124M with AdamW. It uses only Numo::NArray on the CPU and Cumo on the GPU, with no deep learning framework. The same code runs on both backends.

Each model must match its reference implementation exactly. For the language models, that means the same token sequence as llm.c, llama2.c, mamba.c or transformers. Training uses the tolerances of llm.c itself.

The tables below compare Cumo with CuPy and PyTorch ports of the same models. They were measured with cumo 0.11.0 on an RTX 5070 Ti Laptop GPU with locked clocks. In one-token decode, Cumo is 3.4 to 3.6 times as fast as CuPy and 1.06 to 1.39 times as fast as PyTorch. Where matrix multiplication dominates (generation without a KV cache, encoders), Cumo runs at 0.95 to 1.05 times the speed of PyTorch. With int8 weights, it is 2.8 to 2.9 times as fast as PyTorch. PyTorch is faster on ResNet-18 and on training, where Cumo reaches 0.64 times its speed.

The project exists to exercise Numo and Cumo on real workloads. Several performance problems found here have been fixed in Cumo itself. The measurement notes under docs/results/ are in Japanese.

## 数字

要約すると、1 トークンずつ生成する decode では Cumo が CuPy の 3.4〜3.6 倍、PyTorch の 1.06〜1.39 倍速い。行列積が支配する条件 (KV キャッシュ無しの生成、encode) では PyTorch の 0.95〜1.05 倍で、ほぼ並ぶ。int8 では PyTorch の 2.8〜2.9 倍。ResNet-18 と学習は PyTorch が速く、学習は 0.64 倍。

同じ重み・同じ手順で Python の 3 実装と並べたもの。5 実装すべてが同一のトークン列を出すことを関門にしてある。

tokens/sec、プロセスごとに best-of-3 の中央値、11 ラウンドの先頭を位置で捨てて 10。Cumo / CuPy / PyTorch / 対照 (Cumo をもう一度) の 4 系列を 1 ラウンド内でインターリーブし、系列の順序をラウンドごとに回転させて直列に走らせた。クロックは `nvidia-smi -lgc 3090` / `-lmc 14001` で固定してある。GPU は RTX 5070 Ti Laptop。この表は 2026-09-27 に cumo 0.11.0 で測り直したもので、20 条件すべてで対照が 1 をまたいでいる。1 プロセスが 2 秒以上続けて回るように、条件ごとに回数を選んである (`bench/three_impl.tsv`)。

比は同じラウンドどうしの比の中央値で、分子はすべて Cumo。1 を超えれば Cumo が速い。後ろの n/10 は、10 ラウンドのうち比が 1 を超えた (Cumo が速かった) 回数。

### GPT-2 124M

tokens/sec (1 秒あたりに生成したトークン数)。大きいほど速い。

| KV キャッシュ | 生成長 | Cumo | CuPy | PyTorch | Cumo / CuPy | Cumo / PyTorch |
|---|---|---|---|---|---|---|
| 有り | 64 | 840.6 | 230.2 | 792.0 | 3.647 10/10 | 1.061 10/10 |
| 有り | 256 | 818.2 | 228.7 | 717.4 | 3.582 10/10 | 1.140 10/10 |
| 無し | 64 | 493.0 | 207.8 | 501.2 | 2.372 10/10 | 0.982 0/10 |
| 無し | 256 | 228.4 | 142.2 | 241.8 | 1.607 10/10 | 0.945 0/10 |

バッチ生成 (KV キャッシュ有り、長さ 256)。tokens/sec は全系列の合計で、大きいほど速い。

| batch | Cumo | CuPy | PyTorch | Cumo / CuPy | Cumo / PyTorch |
|---|---|---|---|---|---|
| 1 | 822.6 | 229.1 | 715.3 | 3.581 10/10 | 1.149 10/10 |
| 8 | 3856.9 | 1705.9 | 3620.4 | 2.263 10/10 | 1.059 10/10 |

経過と内訳は [gpt2-124m.md](docs/results/gpt2-124m.md)。

### Llama 2 110M (stories110M)

tokens/sec (1 秒あたりに生成したトークン数)。大きいほど速い。

| KV キャッシュ | 生成長 | Cumo | CuPy | PyTorch | Cumo / CuPy | Cumo / PyTorch |
|---|---|---|---|---|---|---|
| 有り | 64 | 745.5 | 219.7 | 560.0 | 3.402 5/5 | 1.327 4/4 |
| 有り | 200 | 741.5 | 218.7 | 558.2 | 3.402 11/11 | 1.324 9/9 |
| 無し | 64 | 482.0 | 205.3 | 459.5 | 2.342 10/10 | 1.051 10/10 |
| 無し | 200 | 262.0 | 173.4 | 270.2 | 1.521 10/10 | 0.980 2/10 またぐ |

キャッシュ有りの 2 行は、計測区間のメモリクロックが 14001 MHz だった回だけで出した (20 ラウンド)。同じバッチの中で一部の回が約 597 に落ちて値が 2 群に分かれ、中央値が群の間に来てしまうためである。比の後ろは、同じラウンドの両方の系列が 14001 だった組の数。長さ 64 では 9001 のままでも高い群に入る回があり、2 群を分けているのは段だけではない (未分離)。経過と内訳は [llama2-110m.md](docs/results/llama2-110m.md)。

### Llama 2 110M を int8 (Q8_0) で

キャッシュ有りのみ。全実装が一致するのは 131 トークンまで。

tokens/sec (1 秒あたりに生成したトークン数)。大きいほど速い。

| 生成長 | Cumo | CuPy | PyTorch | Cumo / CuPy | Cumo / PyTorch |
|---|---|---|---|---|---|
| 64 | 443.0 | 62.5 | 154.9 | 7.096 10/10 | 2.866 10/10 |
| 128 | 438.2 | 62.5 | 154.9 | 7.005 10/10 | 2.836 10/10 |

経過と内訳は [llama2-int8.md](docs/results/llama2-int8.md)。

### Mamba 130M

tokens/sec (1 秒あたりに生成したトークン数)。大きいほど速い。

| 生成長 | Cumo | CuPy | PyTorch | Cumo / CuPy | Cumo / PyTorch |
|---|---|---|---|---|---|
| 64 | 520.9 | 152.5 | 375.1 | 3.417 10/10 | 1.390 10/10 |
| 200 | 517.0 | 152.2 | 374.3 | 3.388 10/10 | 1.376 10/10 |

1 トークン 557 本。経過と内訳は [mamba-130m.md](docs/results/mamba-130m.md)。

### Switch Transformer base-8 (MoE)

tokens/sec。decode は 1 秒あたりに生成したトークン数、encode は 1 秒あたりに encoder を通した入力トークン数。大きいほど速い。

| 条件 | Cumo | CuPy | PyTorch | Cumo / CuPy | Cumo / PyTorch |
|---|---|---|---|---|---|
| decode 72 トークン | 174.0 | 49.9 | 144.8 | 3.493 10/10 | 1.207 10/10 |
| encode 2048 トークン | 2878.6 | 2772.0 | 2833.6 | 1.037 10/10 | 1.018 9/10 またぐ |

経過と内訳は [switch-base-8.md](docs/results/switch-base-8.md)。

### Whisper tiny (音声)

decode は 1 秒あたりに生成したトークン数、encode は 1 秒あたりに encoder を通した音声フレームの位置数 (30 秒の音声が 1500 位置)。大きいほど速い。

| 条件 | Cumo | CuPy | PyTorch | Cumo / CuPy | Cumo / PyTorch |
|---|---|---|---|---|---|
| decode (tokens/sec) | 843.9 | 234.1 | 708.8 | 3.605 10/10 | 1.191 10/10 |
| encode (positions/sec) | 259,190 | 182,376 | 268,945 | 1.421 10/10 | 0.963 0/10 |

メル (3000 フレーム) はこのバッチに入れていない。経過と内訳は [whisper-tiny.md](docs/results/whisper-tiny.md)。

### ResNet-18 (2 次元の畳み込み)

枚/sec (1 秒あたりに分類した画像の枚数)。大きいほど速い。カーネル/pass は 1 回の forward で GPU に投入するカーネルの本数。

| 綴り | Cumo | CuPy | Cumo / CuPy | カーネル/pass (Cumo) |
|---|---|---|---|---|
| `shift` | 683.4 | 430.6 | 1.588 10/10 | 580 |
| `unfold` | 1086.5 | 995.8 | 1.091 10/10 | 384 |
| 綴りの効き (`unfold` / `shift`、中央値の比) | 1.59 倍 | 2.31 倍 | | |
| 対照 (Cumo `unfold` をもう一度) | 1085.9 | | またぐ | |

cuDNN の腕は別のバッチ (6 条件、13 ラウンドの先頭を捨てて 12) なので、上の表とまたいで比を取らないこと。CuPy 14.2 は cuDNN の畳み込みを公開していないので、この表の相手は PyTorch。単位は同じく枚/sec。cumo 0.11.0 では cuDNN の作業領域の上限が既定で 128 MiB で、単精度をテンソルコア (TF32) に載せるのは `CUMO_ALLOW_TF32=1` のときだけである。

| 条件 | 枚/sec | カーネル/pass | |
|---|---|---|---|
| Cumo `unfold` | 1085.8 | 384 | |
| Cumo `cudnn` (既定、fp32) | 1930.2 | 178 | `unfold` の 1.78 倍 |
| Cumo `cudnn` (`CUMO_ALLOW_TF32=1`) | 2080.2 | 216〜217 | `unfold` の 1.91 倍 |
| PyTorch (fp32) | 2320.6 | | Cumo `cudnn` (fp32) の 1.21 倍 |
| PyTorch (TF32、既定) | 2855.7 | | Cumo `cudnn` (TF32) の 1.37 倍 |
| 対照 (Cumo `cudnn` 既定) | 1930.7 | | またぐ |

TF32 ありのカーネル数が整数にならないのは、cuDNN のアルゴリズムの探索がプロセスごとに違う組を選ぶためである。

経過と内訳は [resnet-18.md](docs/results/resnet-18.md)。

### サンプリング (温度 / top-k / top-p)

GPT-2 124M の decode。上の 3 行は 1 トークンあたりに GPU に投入するカーネルの本数 (少ないほど軽い)、tokens/sec は大きいほど速い。最後の行は tokens/sec の比で、1 を超えるほどその条件が遅い。

| 1 トークンあたり | 貪欲法 | top_k 50 | top_p 0.9 | 両方 |
|---|---|---|---|---|
| `sort` (コピー込み) | 0 | 8 | 8 | 8 |
| `cumsum` (CUB) | 0 | 2 | 4 | 4 |
| カーネル合計 | 234 | 262 | 276 | 281 |
| tokens/sec (長さ 900) | 745.3 | 714.2 | 704.6 | 699.7 |
| 貪欲法 / この条件 | — | 1.0431 10/10 | 1.0583 10/10 | 1.0656 10/10 |

経過と内訳は [gpt2-124m.md](docs/results/gpt2-124m.md)。

### GPT-2 124M の学習 (backward + AdamW)

関門は llm.c の `test_gpt2.c` の許容誤差 (勾配 2e-2、損失 1e-2)。表の値は llm.c の参照値との差の最大で、小さいほど参照に近い。

| | 勾配の最大 max\|d\| (関門 B) | 10 ステップの最大 \|d\| (関門 C) |
|---|---|---|
| Numo (CPU) | 1.244e-02 | 1.48e-03 |
| Cumo (GPU) | 6.801e-04 | 1.38e-04 |

Cumo と PyTorch を同じバッチでインターリーブした区間別 (11 ラウンドの先頭を位置で捨てて 10 ラウンド、対照つき)。1 秒あたりにその区間を通せる回数で、合計の行が steps/sec。大きいほど速い。本/step は 1 ステップで GPU に投入するカーネルの本数。測った s/step の逆数で、比も時間の比の逆数から出している。区間の時間は、`--stop-after` で forward、backward、update のそれぞれで止めた 3 本の差である。

| 区間 | 本/step | Cumo | PyTorch | Cumo / PyTorch |
|---|---|---|---|---|
| forward | 331 | 98.3 | 125.8 | 0.80 0/10 (幅が 18% で値は読めない) |
| backward | 1755 | 35.8 | 56.2 | 0.63 0/10 |
| update (AdamW) | 2861 | 29.7 | 50.3 | 0.59 0/10 |
| 合計 | 4947 | 14.0 | 21.9 | 0.64 0/10 |

経過と内訳は [gpt2-124m.md](docs/results/gpt2-124m.md)。

### 1 トークンあたりのカーネル数

decode で 1 トークンあたりに GPU に投入するカーネルの本数 (nsys、`--length 128` と `--length 64` の差 ÷ 64)。少ないほど起動の費用が小さい。

| | 当初 | 現在 | PyTorch (参考) |
|---|---|---|---|
| GPT-2 124M | 640 | 234 | 196 (fp32) / 148 (fp16) |
| Llama 2 110M | 891 | 366 | — |
| Mamba 130M | 1115 | 557 | 1302 |
| Switch base-8 | 2637 | 1533 | — |
| Whisper tiny | — | 293 | — |

計測の作法そのものは [AGENTS.md](AGENTS.md) に、その則がどの測定から来たかは [docs/measurement-cases.md](docs/measurement-cases.md) にある。

条件・手順・外れ値の扱い・そこに至る過程はすべて [docs/results/](docs/results/) にある。[gpt2-124m.md](docs/results/gpt2-124m.md)、[llama2-110m.md](docs/results/llama2-110m.md)、[mamba-130m.md](docs/results/mamba-130m.md)、[switch-base-8.md](docs/results/switch-base-8.md)、[whisper-tiny.md](docs/results/whisper-tiny.md)、[llama2-int8.md](docs/results/llama2-int8.md)、[resnet-18.md](docs/results/resnet-18.md) がモデルごとの記録で、採用しなかった変更とその理由も同じ場所に残してある。

## 計測環境

2026-09-27 時点の開発機。上の表はすべて、rubygems から入れたリリース版の cumo 0.11.0 で測った。

| | |
|---|---|
| 機体 | ASUS ROG NUC 2025 (`NUC15JNKU9X7`)。ラップトップ GPU を積んだミニ PC |
| CPU | Intel Core Ultra 9 275HX (24 スレッド) |
| メモリ | 62 GiB (`free` の total) |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop (Blackwell, sm_120)、VRAM 12 GB、電力上限 100 W (最大 140 W) |
| クロック | 計測中は `nvidia-smi -lgc 3090` / `-lmc 14001` で固定 (`clocks.max.sm` 3090 MHz、`clocks.max.mem` 14001 MHz) |
| OS | CachyOS (Linux 7.2.7-1-cachyos) |
| NVIDIA ドライバ | 615.71.09 |
| CUDA | 13.4 (nvcc V13.4.92) |
| cuDNN | 9.26.0 |
| コンパイラ | GCC 16.2.1 |
| Ruby | 4.0.7 |
| Numo | numo-narray-alt 0.11.2、numo-linalg-alt 0.10.1 (同梱の OpenBLAS 0.3.34) |
| Cumo | 0.11.0 (`CUMO_NVCC_GENERATE_CODE=arch=compute_120,code=sm_120` で `gem install`) |
| Python | 3.14.7、NumPy 2.5.3、CuPy 14.2.0、PyTorch 2.13.0+cu130 (同梱の cuDNN 9.20) |
| プロファイラ | Nsight Systems 2026.3.2 |

GPU の性質 (メモリクロックの段、帯域、電力) は [docs/machine.md](docs/machine.md) にある。同じ表でも段が違えば 1.4 倍動くので、別の機械の数字と並べるときはそちらを先に読むこと。

## 現在の状態

進め方は [PLAN-gpt2.md](docs/plans/PLAN-gpt2.md)、[PLAN-llama2.md](docs/plans/PLAN-llama2.md)、[PLAN-mamba.md](docs/plans/PLAN-mamba.md)、[PLAN-switch.md](docs/plans/PLAN-switch.md)、[PLAN-whisper.md](docs/plans/PLAN-whisper.md)、[PLAN-batch.md](docs/plans/PLAN-batch.md)、[PLAN-sampling.md](docs/plans/PLAN-sampling.md)、[PLAN-training.md](docs/plans/PLAN-training.md)、[PLAN-conv2d.md](docs/plans/PLAN-conv2d.md) に従い、前の段階の受け入れテストが通るまで次の段階のコードは書かない。

| モデル | 段階 | 状態 |
|---|---|---|
| GPT-2 124M | 重みの読み込み / フォワード / 生成 / KV キャッシュ | 完了 (Numo と Cumo が llm.c と同一のトークン列) |
| Llama 2 110M | 同上 | 完了 (Numo と Cumo が llama2.c と同一のトークン列) |
| Llama 2 110M (int8) | decode のみ | 完了 (Numo と Cumo が一致、runq.c と 131 トークン一致) |
| Mamba 130M | 重みの読み込み / フォワード / 生成 | 完了 (Numo と Cumo が mamba.c と同一のトークン列) |
| Switch base-8 (MoE) | 重みの読み込み / encoder / decoder と生成 / 計測 | 完了 (Numo と Cumo が transformers と同一のトークン列。C 参照が無いので関門を作り直した) |
| Whisper tiny (音声) | 重みの読み込み / encoder / decoder と生成 / メル / 計測 / 3 実装の比較 | 完了 (同上。畳み込みとメルスペクトログラムが新しい) |
| GPT-2 124M (学習) | 勾配リーダ / op ごとの backward / モデル全体の backward / AdamW / 計測 | 完了 (Numo と Cumo が llm.c の関門 B・C を通る。完全一致ではなく llm.c の許容誤差) |
| ResNet-18 (2 次元の畳み込み) | 重みと参照 / Conv2d / pooling とブロック / モデル全体 / 計測 | 完了 (16 枚のクラス番号が transformers と完全一致。生成しないモデルなので関門が違う) |

次に何を作るかの候補は [docs/idea.md](docs/idea.md) にある。モデルとは限らない — バッチ生成のように、既にあるモデルへ足すほうが安く広く踏めることもある。

## 実行手順

```
bundle install                                # Numo だけ (CPU)
bundle config set --local with gpu            # Cumo も入れるとき (CUDA toolkit が要る)
CUMO_NVCC_GENERATE_CODE=arch=compute_120,code=sm_120 bundle install

rake download                                 # 6 モデルの重みとトークナイザ (Ruby と curl だけで済む)
rake prepare                                  # 変換と参照値まで作り、テストが読むものを全部揃える

rake test                                     # Numo (CPU)
GPU=1 rake test                               # Cumo (GPU)

ruby script/gpt2_generate.rb --length 256          # GPT-2、EOT 1 個から貪欲法で生成
GPU=1 ruby script/llama2_generate.rb stories110M --length 200
GPU=1 ruby script/llama2_generate.rb stories110M_q80 --length 128   # int8
GPU=1 ruby script/mamba_generate.rb --length 256
GPU=1 ruby script/switch_generate.rb --length 72
GPU=1 ruby script/whisper_generate.rb              # 参照から取ったメルを書き起こす
GPU=1 ruby script/resnet_classify.rb --spelling cudnn   # ResNet-18 で 16 枚を分類
GPU=1 ruby script/gpt2_train.rb                    # GPT-2 を AdamW で 10 ステップ
```

cumo は Gemfile の任意のグループ `gpu` に入れてあり、素の `bundle install` では入らない。`CUMO_NVCC_GENERATE_CODE` は初回起動の JIT を避けるためのもので、sm_120 以外の GPU では値を読み替える。

`rake prepare` は Python と C コンパイラを使う。Mamba と Switch は配布された重みを変換し、Llama 2 と Mamba の参照値は C の参照実装から、Switch・Whisper・ResNet-18 の参照値は transformers から取る。Python は `python/.venv` を見る (`PYTHON=...` で差し替えられる) ので、先に `python/requirements.txt` を入れておく (torch だけは CUDA の版に合わせて別に入れる。手順はファイルの中にある)。揃ったものは作り直さないので何度叩いてもよく、`rake prepare:switch` のようにモデルごとにも呼べる。data/ は全部で約 7 GB になる。

BPE エンコーダは実装していないので、プロンプトはトークン id で渡す (`--tokens 15496,11,995`)。温度・top-k・top-p も入っている (`--top-k 50 --top-p 0.9 --seed 42`)。乱数はホストの `Random` から引くので、両バックエンドが同じ列を出す。

ResNet-18 の `cudnn` は、cumo 0.10.0 からワークスペースの上限が既定 128 MiB になり、何も立てなくても良いアルゴリズムが選ばれる。0.9.0 までは既定が 8 MiB なので `CUMO_CUDNN_MAX_WORKSPACE_SIZE=268435456` を立てる (上げると 1.50 倍)。単精度をテンソルコア (TF32) に載せるのは `CUMO_ALLOW_TF32=1` のときだけで、速くなる代わりに logits が 2 桁動く ([resnet-18.md](docs/results/resnet-18.md))。

必要なものは Ruby (4.0.7 で検証)、`numo-narray-alt`、`numo-linalg-alt` (無くても動くが同じ GEMM が 27 倍遅くなる)、`test-unit`、`rake`。GPU で動かすなら加えて `cumo` と CUDA toolkit。int8 の数字を再現するなら cumo は 0.9.0 以降が要る ([llama2-int8.md](docs/results/llama2-int8.md))。GPU の無い環境でも全テストが通る状態を保っている。

表の数字を測った手順 (Python 側の準備、クロックの固定、3 実装を並べるバッチ `bench/run.sh`、表ごとのコマンド) は [docs/method.md](docs/method.md) の「再現方法」にある。

より詳しい使い方、環境変数、内部の約束事は次にある。

| 読みたいもの | 場所 |
|---|---|
| 計測の条件と手順、表を再現するコマンド | [docs/method.md](docs/method.md) |
| この機械の癖 (クロックの段、帯域、nsys) | [docs/machine.md](docs/machine.md) |
| 重み・トークナイザのファイル形式 (出典つき) | [docs/checkpoint-format-gpt2.md](docs/checkpoint-format-gpt2.md)、[docs/tokenizer-format-gpt2.md](docs/tokenizer-format-gpt2.md)、[同 llama2](docs/checkpoint-format-llama2.md) |
| cumo の未対応の問題 | [docs/cumo-issues.md](docs/cumo-issues.md) |
| cumo で踏んだ問題の経緯 (解決済みを含む) | [docs/cumo-history.md](docs/cumo-history.md) |
| Numo と Cumo の差異、計測とテストの作法 | [AGENTS.md](AGENTS.md) |

## 構成

```
lib/narray_llm/          共有の部品 (backend / ops / generator / sampler / kv_cache /
                         safetensors / backward / adam_w / profiler) と models/ の 6 モデル
                         (gpt2 / llama2 / mamba / switch / whisper / resnet)
script/                  取得・フォワード比較・生成・学習・分類のランナー
test/                    各段階の受け入れテストと部品のユニットテスト
python/                  NumPy / CuPy / PyTorch の比較実装とベンチのドライバ
bench/                   3 実装をインターリーブして回すバッチと集計、条件の一覧
docs/                    形式の仕様、計測結果、機械の特性、段階ごとの計画 (plans/)
data/                    取得した重み (.bin と safetensors) の置き場 (git 管理外)
```

コードは `Numo::` / `Cumo::` を直接書かず `XM` 定数を経由し、フォワードパスのテンソルは `XF` 定数で作る。トークン id のように「デバイスに置くと同期を招く」配列は `HM` (常に Numo) 側に固定している。

図の元データは [docs/architecture.archify.json](docs/architecture.archify.json) で、[archify](https://github.com/tt-a1i/archify) が JSON を検証してから PNG に落としている。

## 謝辞

C の単一ファイル参照に依っている。GPT-2 と学習は Andrej Karpathy の [llm.c](https://github.com/karpathy/llm.c)、Llama 2 と int8 は同じく [llama2.c](https://github.com/karpathy/llama2.c) (どちらも MIT)、Mamba は kroggen の [mamba.c](https://github.com/kroggen/mamba.c) (README に MIT と記載)。重み・参照値・ファイル形式をそのまま使い、関門もそれらの出力に置いている。

C 参照が無いモデルは transformers を参照にした。Switch base-8、Whisper tiny、ResNet-18 の 3 つで、中間活性は再実装ではなく実モデルへの forward hook で取っているので、参照が自前の思い込みからずれない。

重みは Hugging Face の配布をそのまま読む — [google/switch-base-8](https://huggingface.co/google/switch-base-8)、[openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)、[state-spaces/mamba-130m](https://huggingface.co/state-spaces/mamba-130m)、[microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18) (いずれも Apache-2.0)。

ResNet-18 の関門に使う画像は [imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images) から取る。参照を作るときに取得するだけで、このリポジトリには含めていない。

速度の比較対象は CuPy と PyTorch。同じ形の移植を `python/` に置いてあり、どちらも「勝つため」ではなく、出た差が cumo に固有かどうかを分けるために並べている。

そして [Numo::NArray](https://github.com/ruby-numo/numo-narray) と [Cumo](https://github.com/sonots/cumo) — このプロジェクトはそれらを実使用で踏むために書いている。

## ライセンス

コードと文書は [MIT ライセンス](LICENSE)。モデルの重み・参照値・画像はこのリポジトリに含めていないので、取得したものはそれぞれの配布元のライセンスに従う (上の謝辞に挙げたとおり)。llama2.c の `run.c` と mamba.c も取得スクリプトが `vendor/` に置くだけで、追跡していない。
