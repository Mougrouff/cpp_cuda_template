#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <float.h>
#include <stdio.h>

__global__ void __kernel__multiply_array(float* array, float value, int num_kernels){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < num_kernels; index += blockDim.x*gridDim.x){
        array[index] *= value;
    }
}

void __cuda__multiply_array(float* array, const float value, const size_t size, const bool allocated){

    float * cuda_array=nullptr;
    int blockSize=512;

    if(!allocated){
        cudaMalloc(&cuda_array, size * sizeof(float));
        cudaMemcpy(cuda_array, array, size * sizeof(float), cudaMemcpyHostToDevice);
    }
    else{
        cuda_array=array;
    }

    __kernel__multiply_array <<<(size + blockSize - 1) / blockSize,
  	  blockSize, 0/*, get_cuda_stream() */>>>
  		(cuda_array, value, size);

    cudaDeviceSynchronize();

    if(!allocated){
        cudaMemcpy(array, cuda_array, size * sizeof(float), cudaMemcpyDeviceToHost);
    }
}
