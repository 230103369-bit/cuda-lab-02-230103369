
import numpy as np
from numba import cuda

THREADS = (16, 16)


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)

    if row < rows and col < cols:
        if row == 0 or col == 0 or row == rows - 1 or col == cols - 1:
            d_out[row, col] = 0.0
        else:
            d_out[row, col] = (
                -d_in[row - 1, col - 1]
                + d_in[row - 1, col + 1]
                - 2.0 * d_in[row, col - 1]
                + 2.0 * d_in[row, col + 1]
                - d_in[row + 1, col - 1]
                + d_in[row + 1, col + 1]
            )


def run_sobel(h_img):
    h_img = np.asarray(h_img, dtype=np.float32)
    rows, cols = h_img.shape

    if rows == 0 or cols == 0:
        return np.zeros_like(h_img)

    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array((rows, cols), dtype=np.float32)

    blocks = (
        (cols + THREADS[0] - 1) // THREADS[0],
        (rows + THREADS[1] - 1) // THREADS[1]
    )

    sobel_x_kernel[blocks, THREADS](d_in, d_out, rows, cols)
    cuda.synchronize()

    return d_out.copy_to_host()


def main():
    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    rows, cols = 2048, 2048
    np.random.seed(42)

    img = np.random.rand(rows, cols).astype(np.float32)
    result = run_sobel(img)

    expected = np.zeros_like(img)
    expected[1:-1, 1:-1] = (
        -img[:-2, :-2]
        + img[:-2, 2:]
        - 2.0 * img[1:-1, :-2]
        + 2.0 * img[1:-1, 2:]
        - img[2:, :-2]
        + img[2:, 2:]
    )

    max_delta = np.max(np.abs(result - expected))

    assert np.allclose(result, expected, atol=1e-4)

    print("GPU:", cuda.get_current_device().name)
    print("Image Size:", rows, "x", cols)
    print("Threads Per Block:", THREADS)
    print(f"Maximum Difference: {max_delta:.10f}")
    print("TASK 4 PASSED")


if __name__ == "__main__":
    main()
