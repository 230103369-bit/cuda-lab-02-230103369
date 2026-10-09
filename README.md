# CUDA Lab 02

Student ID: 230103369

GPU: NVIDIA Tesla T4

Environment: Google Colab

Language: Python with Numba CUDA

## Task 1 - Warp Divergence

In this task I tested three different CUDA kernels to see how warp divergence affects GPU performance.

I used 1048576 elements, 256 threads per block and 1000 iterations.

Results:

| Kernel | Average (ms) | Std Dev (ms) |
|---|---|---|
| A - Uniform | 25.1344 | 4.8253 |
| B - Full Divergence | 94.5377 | 2.2213 |
| C - Warp Aligned | 46.8113 | 0.0153 |

B / A = 3.76x

B / C = 2.02x

Kernel A was the fastest because all threads were doing the same operations.

Kernel B was much slower because threads inside one warp were going through different branches.

Kernel C was faster than B because the branches were aligned with warps, so there was less divergence.

From this test I understood that warp divergence can make GPU code much slower.

## Task 2 - 1D Stencil

In this task I implemented a 1D stencil using CUDA.

Each element is calculated using its left and right neighbors. For the first and last elements I used boundary clamping.

Array size: 10007

Threads per block: 256

Blocks: 40

Maximum difference: 0.0000000596

Result: PASSED

I compared the GPU output with CPU calculations and the results were almost identical.

## Task 3 - Grid Stride Loop

Here I used grid stride loop to process a large array.

Instead of creating one thread for every element, each thread processes multiple elements.

Array size: 16777216

Threads per block: 256

Blocks: 64

Total threads: 16384

Elements per thread: 1024

Factor: 4.25

Maximum difference: 0.0000000000

Result: PASSED

All elements were multiplied correctly. Grid stride loop is useful when we have more data than GPU threads.

## Task 4 - Sobel 2D

In this task I implemented Sobel X filter using CUDA.

It calculates horizontal gradients in an image using neighboring pixels.

Image size: 2048 x 2048

Threads per block: 16 x 16

Border pixels: 0

Maximum difference: 0.0000004768

Result: PASSED

I compared GPU results with CPU implementation. The difference was very small and the test passed.

## Verification

After completing all tasks I ran verify_submission.py on Google Colab with Tesla T4 GPU.

Results:

Task 2 - PASS

Task 3 - PASS

Task 4 - PASS

Verification successful.

Official Submission Token:

09C4913EEF04C4DB0F57

## Files

task1_divergence.py - Warp divergence experiment

task2_stencil_1d.py - 1D stencil

task3_grid_stride.py - Grid stride loop

task4_sobel_2d.py - Sobel X filter

verify_submission.py - Verification script

## How to run

I used Google Colab with T4 GPU for all tasks.

To run the files:

```bash
python task1_divergence.py
python task2_stencil_1d.py
python task3_grid_stride.py
python task4_sobel_2d.py
```

For verification:

```bash
python verify_submission.py
```

## Conclusion

In this lab I learned more about CUDA programming and how GPU threads work.

The first task showed how warp divergence affects performance. In other tasks I worked with memory boundaries, grid stride loops and 2D image processing.

All four tasks were completed and the verification script passed.
