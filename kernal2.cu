#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__device__ int mandel(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;
        float new_re = (z_re * z_re) - (z_im * z_im);
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandel_kernel(float lower_x, float lower_y, float step_x, float step_y, int *img, int pitch, int res_x, int res_y, int max_iterations) {
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    if (thisX < res_x && thisY < res_y) {
        float x = lower_x + thisX * step_x;
        float y = lower_y + thisY * step_y;
        int *row = (int*)((char*)img + thisY * pitch);
        row[thisX] = mandel(x, y, max_iterations);
    }
}

void host_fe(float upper_x, float upper_y, float lower_x, float lower_y, int *img, int res_x, int res_y, int max_iterations) {
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    int *host_img;
    cudaHostAlloc(&host_img, res_x * res_y * sizeof(int), cudaHostAllocDefault);

    int *device_img;
    size_t pitch;
    cudaMallocPitch(&device_img, &pitch, res_x * sizeof(int), res_y);

    dim3 threadsPerBlock(32, 32);  // 增加线程块大小
    dim3 numBlocks((res_x + threadsPerBlock.x - 1) / threadsPerBlock.x, (res_y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    mandel_kernel<<<numBlocks, threadsPerBlock>>>(lower_x, lower_y, step_x, step_y, device_img, pitch, res_x, res_y, max_iterations);

    cudaMemcpy2D(host_img, res_x * sizeof(int), device_img, pitch, res_x * sizeof(int), res_y, cudaMemcpyDeviceToHost);

    // Copy host_img to img
    for (int i = 0; i < res_x * res_y; ++i) {
        img[i] = host_img[i];
    }

    cudaFreeHost(host_img);
    cudaFree(device_img);
}
