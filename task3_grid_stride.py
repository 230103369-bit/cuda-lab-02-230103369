
import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 64


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, N, stride):
        d_arr[i] *= factor


def run_grid_stride(h_arr, factor):
    N = h_arr.size

    if N == 0:
        return h_arr.copy()

    d_arr = cuda.to_device(h_arr)

    grid_stride_scale_kernel[
        BLOCKS_PER_GRID,
        THREADS_PER_BLOCK
    ](d_arr, factor, N)

    cuda.synchronize()

    return d_arr.copy_to_host()


def main():
    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    N = 16_777_216
    factor = 4.25

    h_arr = np.ones(N, dtype=np.float32)

    result = run_grid_stride(h_arr, factor)

    expected = h_arr * factor
    max_delta = np.max(np.abs(result - expected))

    assert np.allclose(result, expected, atol=1e-4)

    print("GPU:", cuda.get_current_device().name)
    print("Array Size:", N)
    print("Threads Per Block:", THREADS_PER_BLOCK)
    print("Blocks Per Grid:", BLOCKS_PER_GRID)
    print("Total Threads:", THREADS_PER_BLOCK * BLOCKS_PER_GRID)
    print("Elements Per Thread:", N // (THREADS_PER_BLOCK * BLOCKS_PER_GRID))
    print(f"Maximum Difference: {max_delta:.10f}")
    print("TASK 3 PASSED")


if __name__ == "__main__":
    main()
