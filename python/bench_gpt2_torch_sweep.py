"""Benchmark the PyTorch eager implementation, same protocol as bench_gpt2_sweep.py.

  python/.venv/bin/python python/bench_gpt2_torch_sweep.py            # CPU
  GPU=1 python/.venv/bin/python python/bench_gpt2_torch_sweep.py      # CUDA

This column is the ceiling reference, not a structural mirror: torch is allowed
its fused kernels. What is held constant is the weights, fp32, TF32 off, the
prefill/decode split, greedy decoding, and the timing protocol.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import gpt2_torch as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("NARRAY_LLM_DATA", os.path.join(ROOT, "data"))
FIXTURE = os.path.join(ROOT, "python", "fixtures", "gpt2_124M_greedy.json")
LENGTHS = [64, 256]
REPEAT = int(os.environ.get("REPEAT", 3))
CLOSE_CALL = 1e-3


def best_of(repeat, fn):
    times = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        T.synchronize()
        times.append(time.perf_counter() - started)
    return min(times), times


def elementwise_cost():
    """Same method as bench_gpt2_sweep.py: plain a + b, output allocated every call and
    discarded, warm up 100, best-of-5 over 1000 calls."""
    a = torch.ones(256, dtype=torch.float32, device=T.DEVICE)
    b = torch.ones(256, dtype=torch.float32, device=T.DEVICE)

    def thousand():
        for _ in range(1000):
            a + b

    for _ in range(100):
        a + b
    T.synchronize()
    best, _ = best_of(5, thousand)
    return best / 1000


def check_fixture(model, fixture, length, tokens):
    expected = fixture["sequences"][str(length)]
    got = tokens[1:]
    if got == expected:
        return True, None
    index = next((i for i in range(min(len(got), len(expected))) if got[i] != expected[i]),
                 min(len(got), len(expected)))
    prefix = [fixture["prompt"][0]] + expected[:index]
    cache = model.new_cache()
    logits = model.prefill([prefix[:1]], cache)
    for i in range(1, len(prefix)):
        logits = model.decode(prefix[i], i, cache)
    row = logits.reshape(-1).float().cpu()
    top = torch.topk(row, 2)
    gap = float(top.values[0] - top.values[1])
    detail = (f"位置 {index} で割れた: torch={got[index]} ruby={expected[index]}\n"
              f"    その位置の logits 1 位 {int(top.indices[0])}={float(top.values[0]):.6f}, "
              f"2 位 {int(top.indices[1])}={float(top.values[1]):.6f}, 差={gap:.6e}")
    if gap <= CLOSE_CALL:
        detail += ("\n    差が 1e-3 以下の僅差。SDPA は演算順序が違うので想定内。"
                   "ここ以降の比較は打ち切る。")
    return False, detail


def main():
    backend = "PyTorch eager (CUDA)" if T.GPU else "PyTorch eager (CPU)"
    print("== 環境 ==")
    print(f"python        : {platform.python_version()}")
    print(f"torch         : {torch.__version__}")
    print(f"device        : {T.DEVICE}")
    if T.GPU:
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"CUDA          : {torch.version.cuda}")
    print(f"TF32          : matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
          f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")
    print(f"threads       : {torch.get_num_threads()}")

    with torch.inference_mode():
        load_started = time.perf_counter()
        model = T.Model(os.path.join(DATA_DIR, "gpt2_124M.bin"))
        eot, table = T.load_tokenizer(os.path.join(DATA_DIR, "gpt2_tokenizer.bin"))
        generator = T.Generator(model, eot_token=eot)
        print(f"SDPA backend  : prefill(T=64)={T.sdpa_backend(model, 64, 64)}, "
              f"decode(T=1)={T.sdpa_backend(model, 1, 64)}")
        print(f"重み読み込み  : {time.perf_counter() - load_started:.2f} s (計測には含めない)")
        print(f"メモリ見積もり: 重み {model.parameter_bytes() / 2**20:.1f} MiB + "
              f"KV キャッシュ {T.KVCache.bytes_for(model.config.num_layers, model.config.max_seq_len, model.config.channels) / 2**20:.1f} MiB")
        print(f"1 演算あたりの固定費 (a+b, 256 要素): {elementwise_cost() * 1e6:.2f} us")
        print()

        fixture = json.load(open(FIXTURE))
        print("== 生成トークン列の一致 (Ruby の fixture との完全一致) ==")
        for length in LENGTHS:
            tokens = generator.generate([eot], max_new_tokens=length, cache=True)
            ok, detail = check_fixture(model, fixture, length, tokens)
            print(f"length={length}: {'一致' if ok else 'NG'}")
            if detail:
                print(f"    {detail}")
        print(f"サンプル: {T.decode_text(table, tokens[1:])[:70]!r}...")
        print()

        print(f"== 速度 (best-of-{REPEAT}, ウォームアップ 1 回) ==")
        print(f"{'cache':<7} {'length':>7} {'seconds':>10} {'tokens/sec':>12}   各回")
        for cache in (True, False):
            for length in LENGTHS:
                generator.generate([eot], max_new_tokens=2, cache=cache)
                T.synchronize()
                best, times = best_of(REPEAT, lambda c=cache, n=length:
                                      generator.generate([eot], max_new_tokens=n, cache=c))
                print(f"{'ON' if cache else 'OFF':<7} {length:>7} {best:>10.3f} "
                      f"{length / best:>12.2f}   " + ", ".join(f"{t:.3f}" for t in times))


if __name__ == "__main__":
    main()
