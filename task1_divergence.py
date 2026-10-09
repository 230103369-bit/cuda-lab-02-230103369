import numpy as np
import time
from numba import cuda

# ==========================================
# CUDA LAB 02 - TASK 1
# Warp Divergence Microbenchmark
# ==========================================

N = 1_048_576
ITERATIONS = 1000
THREADS_PER_BLOCK = 256
TRIALS = 10


# Path 1: Multiply-Accumulate
@cuda.jit
def kernel_a_uniform(d_in, d_out):
    idx = cuda.grid(1)

    if idx < d_in.size:
        value = d_in[idx]

        for j in range(ITERATIONS):
            value = value * 1.00001 + 0.00001

        d_out[idx] = value


# Kernel B: Full Divergence
@cuda.jit
def kernel_b_divergent(d_in, d_out):
    idx = cuda.grid(1)

    if idx < d_in.size:
        value = d_in[idx]

        if idx % 2 == 0:
            for j in range(ITERATIONS):
                value = value * 1.00001 + 0.00001
        else:
            for j in range(ITERATIONS):
                value = (value - 0.00001) / 1.00001

        d_out[idx] = value


# Kernel C: Warp-Aligned Branching
@cuda.jit
def kernel_c_warp_aligned(d_in, d_out):
    idx = cuda.grid(1)

    if idx < d_in.size:
        value = d_in[idx]
        warp_id = idx // 32

        if warp_id % 2 == 0:
            for j in range(ITERATIONS):
                value = value * 1.00001 + 0.00001
        else:
            for j in range(ITERATIONS):
                value = (value - 0.00001) / 1.00001

        d_out[idx] = value


# ==========================================
# Benchmark
# ==========================================

def benchmark(kernel, d_in, d_out, blocks):
    # Warm-up
    kernel[blocks, THREADS_PER_BLOCK](d_in, d_out)
    cuda.synchronize()

    times = []

    for _ in range(TRIALS):
        start = time.perf_counter()

        kernel[blocks, THREADS_PER_BLOCK](d_in, d_out)
        cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return np.mean(times), np.std(times)


def main():
    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is not available!")

    device = cuda.get_current_device()

    print("=" * 60)
    print("CUDA LAB 02 - TASK 1")
    print("WARP DIVERGENCE MICROBENCHMARK")
    print("=" * 60)

    print(f"GPU: {device.name}")
    print(f"Compute Capability: {device.compute_capability}")
    print(f"Array Size: {N:,}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Trials: {TRIALS}")

    np.random.seed(42)
    h_in = np.random.rand(N).astype(np.float32)

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(d_in)

    blocks = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    print(f"Threads Per Block: {THREADS_PER_BLOCK}")
    print(f"Blocks Per Grid: {blocks}")
    print("=" * 60)

    results = {}

    kernels = [
        ("A - Uniform", kernel_a_uniform),
        ("B - Full Divergence", kernel_b_divergent),
        ("C - Warp-Aligned", kernel_c_warp_aligned),
    ]

    for name, kernel in kernels:
        avg, std = benchmark(kernel, d_in, d_out, blocks)

        results[name] = avg

        print(f"{name}")
        print(f"  Average: {avg:.4f} ms")
        print(f"  Std Dev: {std:.4f} ms")
        print()

    a = results["A - Uniform"]
    b = results["B - Full Divergence"]
    c = results["C - Warp-Aligned"]

    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    print(f"Kernel A: {a:.4f} ms")
    print(f"Kernel B: {b:.4f} ms")
    print(f"Kernel C: {c:.4f} ms")

    print()
    print(f"B / A Ratio: {b/a:.2f}x")
    print(f"B / C Ratio: {b/c:.2f}x")

    print()
    print("TASK 1 BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
