"""Benchmark the NumPy / CuPy GPT-2 against the Ruby implementation.

Same protocol as script/gpt2_generate.rb on the Ruby side: one EOT token, greedy
decoding, lengths 64 and 256, best-of-3 with one warm-up, no synchronization
inside the timed region other than the one that ends it.

  python/.venv/bin/python python/bench_gpt2_sweep.py            # NumPy
  GPU=1 python/.venv/bin/python python/bench_gpt2_sweep.py      # CuPy

Everything that could make the comparison unfair is printed rather than
assumed: the BLAS in use and its thread count, whether TF32 is on, and the
library versions.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# TF32 changes both speed and results, and would break the token match. CuPy
# reads this at import time, so it has to be set before cupy is imported.
os.environ.setdefault("CUPY_TF32", "0")

import numpy as np  # noqa: E402

import gpt2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("NARRAY_LLM_DATA", os.path.join(ROOT, "data"))
FIXTURE = os.path.join(ROOT, "python", "fixtures", "gpt2_124M_greedy.json")
LENGTHS = [64, 256]
REPEAT = int(os.environ.get("REPEAT", 3))
# docs/cumo-history.md: below this the two implementations are picking between
# logits that a different GEMM kernel ordering can reorder.
CLOSE_CALL = 1e-3


def blas_threads():
    try:
        import threadpoolctl
        return [(d.get("internal_api"), d.get("num_threads")) for d in threadpoolctl.threadpool_info()]
    except ImportError:
        return [("threadpoolctl not installed", None)]


def report_environment():
    print("== 環境 ==")
    print(f"python        : {platform.python_version()}")
    print(f"numpy         : {np.__version__}")
    build = np.show_config("dicts").get("Build Dependencies", {}).get("blas", {})
    print(f"numpy BLAS    : {build.get('name')} {build.get('version')}")
    print(f"BLAS threads  : {blas_threads()}")
    print(f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', '(未設定)')} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(未設定)')}")
    if gpt2.GPU:
        import cupy
        from cupy.cuda import cublas
        mode = cublas.getMathMode(cupy.cuda.device.get_cublas_handle())
        print(f"cupy          : {cupy.__version__}")
        print(f"CUDA runtime  : {cupy.cuda.runtime.runtimeGetVersion()}")
        print(f"device        : {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"CUPY_TF32     : {os.environ.get('CUPY_TF32')} "
              f"(cublas math mode={mode}, 0 は FP32 で TF32 無効)")
    print()


def best_of(repeat, fn):
    times = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        gpt2.synchronize()
        times.append(time.perf_counter() - started)
    return min(times), times


def elementwise_cost():
    """Per-operation fixed cost. Plain `a + b`, allocating the output every
    call and discarding it, which is what the Ruby side's `a + b` does. Not
    `add(a, b, out=c)`, which is about 20% cheaper here but has no counterpart
    on the Ruby side, and no fusion or graph capture for the same reason."""
    a = gpt2.xp.ones((256,), dtype=gpt2.xp.float32)
    b = gpt2.xp.ones((256,), dtype=gpt2.xp.float32)

    def thousand():
        for _ in range(1000):
            a + b

    for _ in range(100):
        a + b
    gpt2.synchronize()
    best, _ = best_of(5, thousand)
    return best / 1000


def check_fixture(generator, fixture, length, tokens):
    expected = fixture["sequences"][str(length)]
    got = tokens[1:]
    if got == expected:
        return True, None
    index = next((i for i in range(min(len(got), len(expected))) if got[i] != expected[i]),
                 min(len(got), len(expected)))
    prefix = [fixture["prompt"][0]] + expected[:index]
    logits = replay_logits(generator.model, prefix)
    row = np.asarray(gpt2.xp.asnumpy(logits.reshape(-1)) if gpt2.GPU else logits.reshape(-1))
    top = np.argsort(row)[-2:][::-1]
    gap = float(row[top[0]] - row[top[1]])
    detail = (f"位置 {index} で割れた: python={got[index]} ruby={expected[index]}\n"
              f"    その位置の logits 1 位 {int(top[0])}={row[top[0]]:.6f}, "
              f"2 位 {int(top[1])}={row[top[1]]:.6f}, 差={gap:.6e}")
    if gap <= CLOSE_CALL:
        detail += ("\n    差が 1e-3 以下の僅差。docs/cumo-history.md の「差ではなかったもの」に"
                   "ある GEMM のカーネル切り替えと同根なので、ここ以降の比較は打ち切る。")
    return False, detail


def replay_logits(model, prefix):
    cache = model.new_cache()
    logits = model.prefill([prefix[:1]], cache)
    for i in range(1, len(prefix)):
        logits = model.decode(prefix[i], i, cache)
    return logits


def main():
    backend = "CuPy (GPU)" if gpt2.GPU else "NumPy (CPU)"
    report_environment()

    load_started = time.perf_counter()
    model = gpt2.Model(os.path.join(DATA_DIR, "gpt2_124M.bin"))
    eot, table = gpt2.load_tokenizer(os.path.join(DATA_DIR, "gpt2_tokenizer.bin"))
    generator = gpt2.Generator(model, eot_token=eot)
    print(f"重み読み込み  : {time.perf_counter() - load_started:.2f} s (計測には含めない)")
    print(f"メモリ見積もり: 重み {model.parameter_bytes() / 2**20:.1f} MiB + "
          f"KV キャッシュ {gpt2.KVCache.bytes_for(model.config.num_layers, model.config.max_seq_len, model.config.channels) / 2**20:.1f} MiB")
    print(f"1 演算あたりの固定費 (a+b, 256 要素): {elementwise_cost() * 1e6:.2f} us")
    print()

    fixture = json.load(open(FIXTURE))
    print("== 生成トークン列の一致 (Ruby の fixture との完全一致) ==")
    for length in LENGTHS:
        tokens = generator.generate([eot], max_new_tokens=length, cache=True)
        ok, detail = check_fixture(generator, fixture, length, tokens)
        print(f"length={length}: {'一致' if ok else 'NG'}")
        if detail:
            print(f"    {detail}")
    print(f"サンプル: {gpt2.decode_text(table, tokens[1:])[:70]!r}...")
    print()

    print(f"== 速度 (best-of-{REPEAT}, ウォームアップ 1 回) ==")
    print(f"{'cache':<7} {'length':>7} {'seconds':>10} {'tokens/sec':>12}   各回")
    rows = []
    for cache in (True, False):
        for length in LENGTHS:
            generator.generate([eot], max_new_tokens=2, cache=cache)
            gpt2.synchronize()
            best, times = best_of(REPEAT, lambda c=cache, n=length:
                                  generator.generate([eot], max_new_tokens=n, cache=c))
            rows.append((backend, cache, length, length / best))
            print(f"{'ON' if cache else 'OFF':<7} {length:>7} {best:>10.3f} {length / best:>12.2f}   "
                  + ", ".join(f"{t:.3f}" for t in times))
    print()
    return rows


if __name__ == "__main__":
    main()
