#ifndef CUDA_KERNEL_HEADER
#define CUDA_KERNEL_HEADER

    #ifdef USE_CUDA
    #endif

    void __cuda__multiply_array(float* array, const float value, const size_t size, const bool allocated);
#endif
