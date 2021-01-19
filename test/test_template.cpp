#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include "hello.h"

#ifdef USE_CUDA
    #include "cuda_runtime.h"
    #include "cuda_kernels.h"
#endif



void print_float_array(float* array, int size, const std::string& separator){

    std::cout << "Float array [" << size << "] : " << std::endl << std::endl;
    for(int i=0; i<size; i++)
        std::cout << array[i] << separator;
    std::cout << std::endl;

}

void multiply_array_cpu(float* array, const float value, const size_t size){
    for (size_t i=0; i<size; i++){
        array[i] *= value;
    }
}

float* generate_array(size_t size){

    float* ret = (float*)calloc(size, sizeof(float));

    for (size_t i=0; i<size; i++)
        ret[i] = ((float) rand() / (RAND_MAX));

    return ret;
}

bool are_equals(float* arr_1, float* arr_2, size_t size){
    for(int i=0; i<size; i++){
        if(abs(arr_1[i] - arr_2[i]) > 0.0001)
            return false;
    }

    return true;
}

float* clone(float* src, size_t size){
    float* ret = (float*)calloc(size, sizeof(float));

    for (size_t i=0; i<size; i++)
        ret[i] = src[i];

    return ret;
}

int main(){

    srand(time(NULL));

    const int arr_size = 100;
    float* array_0 = generate_array(arr_size);
    float* array_1 = clone(array_0, arr_size);
    float* array_2 = clone(array_0, arr_size);

    //std::string separator = "\n";

    std::chrono::time_point<std::chrono::system_clock> start;
    std::chrono::time_point<std::chrono::system_clock> end;

    float cpu_time=0;
    float gpu_time=0;
    float gpu_time_noalloc=0;

    start = std::chrono::system_clock::now();
    multiply_array_cpu(array_0, 2.0f, arr_size);
    end = std::chrono::system_clock::now();

    cpu_time = (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;



    start = std::chrono::system_clock::now();
    __cuda__multiply_array(array_1, 2.0f, arr_size, false);
    end = std::chrono::system_clock::now();

    gpu_time = (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;


    float* cuda_arr;
    cudaMalloc(&cuda_arr, arr_size * sizeof(float));
    cudaMemcpy(cuda_arr, array_2, arr_size * sizeof(float), cudaMemcpyHostToDevice);

    start = std::chrono::system_clock::now();
    __cuda__multiply_array(cuda_arr, 2.0f, arr_size, true);
    end = std::chrono::system_clock::now();

    cudaMemcpy(array_2, cuda_arr, arr_size * sizeof(float), cudaMemcpyDeviceToHost);

    gpu_time_noalloc = (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;

    std::cout << "CPU : " << cpu_time << "ms" << std::endl;
    std::cout << "GPU : " << gpu_time << "ms" << std::endl;
    std::cout << "GPU*: " << gpu_time_noalloc << "ms  (*without cudaMemcpy)"  <<std::endl;

    std::cout << "Equals : " << ( are_equals(array_0, array_1, arr_size) && are_equals(array_1, array_2, arr_size) ) << std::endl;

    hello("Hugo");

    free(array_0);
    free(array_1);
    free(array_2);
    return 0;
}
