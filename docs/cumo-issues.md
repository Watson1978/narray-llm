# cumo の未対応の問題

このリポジトリで踏んで、cumo の側でまだ直っていない問題。2026-09-23 に cumo 0.10.0 と master (`ba27577a`) で残っていることを確かめた。解決した問題とこれまでの経緯は [cumo-history.md](cumo-history.md) にある。

## `cumsum` が 8192 要素未満でホストへ同期する

`CUMO_SHOW_WARNING=ON` で次の警告が出る。cumo 自身が `FIXME` と書いている。

```
Warning: FIXME: Method "cumsum" for dtype "sfloat" synchronizes with CPU.
```

`ext/cumo/narray/gen/tmpl/cum.c` は、要素数が `CUMO_CUM_MIN_KERNEL_SIZE` (`ext/cumo/include/cumo/template.h:74` の 8192) 未満だとデバイス全体を同期してからホストのループで計算する。8192 という値の根拠は cumo 側も測っていない。

Switch の MoE で同点の expert を 1 つに絞る走査に使って踏んだ (1 トークンあたり 6 回の同期)。走る軸が短いなら、厳密下三角行列との積で同じ値が出る。Switch はそう書き換えて 6.4% 速くなったが、その差が同期の値段だけかどうかは分けていない。詳細は [cumo-history.md](cumo-history.md) の「`cumsum` はホストへ同期する」と [results/switch-base-8.md](results/switch-base-8.md)。

## NArray を添字にした gather が同期する

`a[narray_index]` のように NArray を添字にすると、添字をホストに読み戻して範囲を検査するので同期する。Ruby の Array を添字にすれば同期しない。

```
Warning: Method "cumo_na_parse_narray_index" for dtype "any" synchronizes with CPU.
```

デバイス上で決まる添字 (MoE の expert 選択、ONNX の Gather) を、デバイスに置いたまま引く方法が無い。Switch は expert 番号をホストに読み戻してから Ruby の Array で添字にしており、疎な層ごとに 1 回同期する。

## 読み戻しが 1 回 18 us かかる

cumo 0.10.0 (#528) で、decode の形の `extract_cpu` が 143.8 us から 18.1 us に縮んだ。それでも 0 ではなく、GPT-2 の decode では読み戻しを外すと 1.6% 速い (712.8 対 724.5 tokens/sec)。このリポジトリはループの中の読み戻しを 1 トークンに 1 回までにしている。詳細は [cumo-history.md](cumo-history.md) の「0.10.0 は decode を 4〜10% 速くする」。

## `ElementwiseKernel` が fp16 と bf16 を受け付けない

`Cumo::CUDA::ElementwiseKernel` に `Cumo::HFloat` か `Cumo::BFloat` を渡すと例外になる。

```
TypeError: a Cumo::HFloat cannot be handed to a kernel
TypeError: a Cumo::BFloat cannot be handed to a kernel
```

GPT-2 の学習で試した融合 AdamW はこれで書いているので、fp16 と bf16 では使えない ([results/gpt2-124m.md](results/gpt2-124m.md) の「AdamW を 1 カーネルに融合した」)。

## 読み書きの両方がストライドのビューになるコピーが CuPy より遅い

ResNet-18 の `unfold` で窓行列を組み立てるコピーが、3x3 の 16 層すべてで CuPy の 1.18〜1.26 倍かかる。読み出しは空間方向にストライドのあるビュー、書き込みは窓行列の列スライスで、Cumo は `cumo_iter_sfloat_store_sfloat_kernel_dim4`、CuPy は `cupy_copy__float32_float32` を使う。

別セッションが同じ形を 4 通りに分けると、両側がストライドの形だけが遅かった (1 タップあたり 0.0494 ms。片側だけストライドなら 0.0368 ms と 0.0260 ms、両側が連続なら 0.0187 ms)。ResNet-18 の `unfold` で Cumo が CuPy に 7.6% 負ける差のうち、54% がこの組み立てである。詳細は [results/resnet-18.md](results/resnet-18.md) の「7.3% の行き先は分かった」。

## 行列積が支配する encode で PyTorch に負ける。出どころは未分離

Whisper tiny の encode (1500 位置) で、PyTorch が Cumo の 1.15 倍速い (README の表では Cumo / PyTorch が 0.869、10 ラウンドすべてで PyTorch が速い)。同じ表で Cumo は CuPy の 1.27 倍速いので、PyTorch だけが速い。

3 実装とも畳み込みを行列積で書いてあり、GEMM は 1 encode あたり 78 本で一致し、Cumo と PyTorch は cutlass のタイルと grid まで同じだった。GPU 時間を 3 組取ると、GEMM の時間は区別できず、GEMM 以外 (Cumo 139 本、PyTorch 99 本) は区別できた。ただし nsys の時間なので、wall の差をそこへ帰属させてはいない。

cumo 側の候補は 4 つ挙がっていて、どれも測っていない。

1. 2 次元 1 本の行列積でも `gemmStridedBatched` を通す (`gen/tmpl/gemm.c`)
2. `CUBLAS_GEMM_DEFAULT` を固定で渡す (PyTorch は cuBLASLt のヒューリスティクスを使う)
3. 呼び出しごとのホスト側の費用 (レイアウトの判定、連続性の検査)
4. ワークスペースやハンドルの設定

詳細は [cumo-history.md](cumo-history.md) の「encode で PyTorch が 15.3% 速い」と [results/whisper-tiny.md](results/whisper-tiny.md)。

## `MemoryPool.used_bytes` が GC の前に多く出る

参照が切れていても、Ruby の GC がまだ回収していないブロックを使用中として数える。Whisper tiny の重みを読んだ直後で 294.9 MiB、`GC.start` の後で 218.9 MiB だった。CuPy の `used_bytes()` や PyTorch の `memory_allocated()` とは同じ名前でも意味が違う。

生きているテンソルの量を知りたいなら `GC.start` の後に `used_bytes` を読み、プールがドライバから取った量なら `total_bytes` を読む。cumo は `cudaMallocManaged` で確保するので、`nvidia-smi` のプロセスごとの値はデバイスに常駐している分しか数えない。詳細は [cumo-history.md](cumo-history.md) の同じ節。
