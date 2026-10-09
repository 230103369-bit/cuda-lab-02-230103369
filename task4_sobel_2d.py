
import numpy as np
import math
from numba import cuda

THREADS = (16, 16)


@cuda.jit
def sobel_kernel(d_in, d_out, height, width):
    x, y = cuda.grid(2)

    if x < width and y < height:
        if x == 0 or y == 0 or x == width - 1 or y == height - 1:
            d_out[y, x] = 0.0
        else:
            gx = (
                -d_in[y - 1, x - 1]
                + d_in[y - 1, x + 1]
                - 2.0 * d_in[y, x - 1]
                + 2.0 * d_in[y, x + 1]
                - d_in[y + 1, x - 1]
                + d_in[y + 1, x + 1]
            )

            gy = (
                -d_in[y - 1, x - 1]
                - 2.0 * d_in[y - 1, x]
                - d_in[y - 1, x + 1]
                + d_in[y + 1, x - 1]
                + 2.0 * d_in[y + 1, x]
                + d_in[y + 1, x + 1]
            )

            d_out[y, x] = math.sqrt(gx * gx + gy * gy)


def run_sobel(h_img):
    height, width = h_img.shape

    if height == 0 or width == 0:
        return np.zeros_like(h_img, dtype=np.float32)

    h_img = np.asarray(h_img, dtype=np.float32)

    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array((height, width), dtype=np.float32)

    blocks = (
        (width + THREADS[0] - 1) // THREADS[0],
        (height + THREADS[1] - 1) // THREADS[1]
    )

    sobel_kernel[blocks, THREADS](d_in, d_out, height, width)
    cuda.synchronize()

    return d_out.copy_to_host()


def cpu_sobel(img):
    height, width = img.shape
    result = np.zeros((height, width), dtype=np.float32)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gx = (
                -img[y - 1, x - 1]
                + img[y - 1, x + 1]
                - 2.0 * img[y, x - 1]
                + 2.0 * img[y, x + 1]
                - img[y + 1, x - 1]
                + img[y + 1, x + 1]
            )

            gy = (
                -img[y - 1, x - 1]
                - 2.0 * img[y - 1, x]
                - img[y - 1, x + 1]
                + img[y + 1, x - 1]
                + 2.0 * img[y + 1, x]
                + img[y + 1, x + 1]
            )

            result[y, x] = np.sqrt(gx * gx + gy * gy)

    return result


def main():
    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    np.random.seed(42)

    height = 256
    width = 256

    h_img = np.random.rand(height, width).astype(np.float32)

    gpu_result = run_sobel(h_img)
    cpu_result = cpu_sobel(h_img)

    max_delta = np.max(np.abs(gpu_result - cpu_result))

    assert np.allclose(gpu_result, cpu_result, atol=1e-4)

    print("GPU:", cuda.get_current_device().name)
    print("Image Size:", height, "x", width)
    print("Threads Per Block:", THREADS)
    print(f"Maximum Difference: {max_delta:.10f}")
    print("TASK 4 PASSED")


if __name__ == "__main__":
    main()
