
import numpy as np
from numba import cuda


# ==========================================
# CUDA LAB 02 - TASK 2
# 1D Boundary Stencil & Halo Protection
# ==========================================

THREADS_PER_BLOCK = 256


# ==========================================
# CUDA Kernel
# ==========================================

@cuda.jit
def stencil_1d(d_in, d_out, N):

    idx = cuda.grid(1)

    if idx < N:

        # Boundary clamping
        left = max(idx - 1, 0)
        right = min(idx + 1, N - 1)

        # 3-point stencil
        d_out[idx] = (
            0.25 * d_in[left]
            + 0.5 * d_in[idx]
            + 0.25 * d_in[right]
        )


# ==========================================
# GPU Host Function
# ==========================================

def run_stencil(h_in):

    N = h_in.size

    if N == 0:
        return np.empty_like(h_in)

    # Transfer data from CPU to GPU
    d_in = cuda.to_device(h_in)

    # Allocate GPU output
    d_out = cuda.device_array_like(d_in)

    # Calculate grid size
    blocks = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    # Launch CUDA Kernel
    stencil_1d[blocks, THREADS_PER_BLOCK](d_in, d_out, N)

    cuda.synchronize()

    # Transfer result back to CPU
    return d_out.copy_to_host()


# ==========================================
# CPU Reference Implementation
# ==========================================

def cpu_stencil(arr):

    padded = np.pad(arr, (1, 1), mode="edge")

    return (
        0.25 * padded[:-2]
        + 0.5 * padded[1:-1]
        + 0.25 * padded[2:]
    )


# ==========================================
# Main Test
# ==========================================

def main():

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is not available!")

    device = cuda.get_current_device()

    print("=" * 60)
    print("CUDA LAB 02 - TASK 2")
    print("1D BOUNDARY STENCIL & HALO PROTECTION")
    print("=" * 60)

    print(f"GPU: {device.name}")
    print(f"Threads Per Block: {THREADS_PER_BLOCK}")

    # Test 1: Small array
    h_small = np.array(
        [10, 20, 30, 40, 50],
        dtype=np.float32
    )

    gpu_small = run_stencil(h_small)
    cpu_small = cpu_stencil(h_small)

    print("\nTEST 1: SMALL ARRAY")

    print("Input:        ", h_small)
    print("GPU Result:   ", gpu_small)
    print("CPU Reference:", cpu_small)

    assert np.allclose(
        gpu_small,
        cpu_small,
        atol=1e-4
    )

    print("[PASS] Small Array Test")

    # Test 2: Assignment verification size
    N = 10007

    np.random.seed(42)

    h_in = np.random.rand(N).astype(np.float32)

    gpu_result = run_stencil(h_in)
    cpu_result = cpu_stencil(h_in)

    max_delta = np.max(
        np.abs(gpu_result - cpu_result)
    )

    passed = np.allclose(
        gpu_result,
        cpu_result,
        atol=1e-4
    )

    print("\nTEST 2: LARGE ARRAY")

    print(f"Array Size: {N}")
    print(f"Threads Per Block: {THREADS_PER_BLOCK}")

    blocks = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    print(f"Blocks Per Grid: {blocks}")
    print(f"Maximum Difference: {max_delta:.10f}")

    if passed:
        print(f"\nTASK 2 PASSED: MAX DELTA = {max_delta:.10f}")
    else:
        print("\nTASK 2 FAILED")
        raise AssertionError("GPU result does not match CPU reference")


if __name__ == "__main__":
    main()
