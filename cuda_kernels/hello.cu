/**
 * @file hello.cu
 * @brief A simple CUDA kernel that prints "Hello, World".
 * @author Vishank Singh, https://github.com/VishankSingh
 */

#include <cuda_runtime.h>
#include <cstdio>


__global__ void helloWorldKernel() {
    int threadId = threadIdx.x;
    printf("Hello, World from CUDA! Thread ID: %d\n", threadId);
}

void launchHelloWorldKernel() {
    // dim3 grid(2);
    // dim3 block(1); // Launching 5 threads in a block
    // // Launch the kernel with 1 block and 5 threads
    // helloWorldKernel<<<grid, block>>>();

    helloWorldKernel<<<1, 2>>>();
    cudaDeviceSynchronize(); // Ensure the kernel execution is complete
}



