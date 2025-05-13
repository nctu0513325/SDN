#include <cstdio>
#include <cstdlib>
#include "mandel.h"

__global__ void mandel_kernel(float lower_x, float lower_y, float step_x, float step_y, int *img, int res_x, int res_y, int max_iterations) {
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    if (thisX < res_x && thisY < res_y) {
        float x = lower_x + thisX * step_x;
        float y = lower_y + thisY * step_y;
        // Compute mandelbrot value and store in img
    }
}

void host_fe(float upper_x, float upper_y, float lower_x, float lower_y, int *img, int res_x, int res_y, int max_iterations) {
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    int *host_img = new int[res_x * res_y];
    int *device_img;
    cudaMalloc(&device_img, res_x * res_y * sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((res_x + threadsPerBlock.x - 1) / threadsPerBlock.x, (res_y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    mandel_kernel<<<numBlocks, threadsPerBlock>>>(lower_x, lower_y, step_x, step_y, device_img, res_x, res_y, max_iterations);

    cudaMemcpy(host_img, device_img, res_x * res_y * sizeof(int), cudaMemcpyDeviceToHost);

    // Copy host_img to img
    for (int i = 0; i < res_x * res_y; ++i) {
        img[i] = host_img[i];
    }

    delete[] host_img;
    cudaFree(device_img);
}
