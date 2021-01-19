#include <iostream>
#include "hello.h"

#ifdef USE_CUDA
    #include "cuda_kernels.h"
#endif

#include "cuda_kernels.h"



void print_float_array(float* array, int size, const std::string& separator){

    std::cout << "Float array [" << size << "] : " << std::endl << std::endl;
    for(int i=0; i<size; i++)
        std::cout << array[i] << separator;
    std::cout << std::endl;

}

int main(){

    float array[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const int tab_size = 10;
    std::string separator = "\n";

    print_float_array(array, tab_size, separator);

    __cuda__multiply_array(array, 2.0f, tab_size, false);

    print_float_array(array, tab_size, separator);

    hello("Hugo");
    return 0;
}
