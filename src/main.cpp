/**
 * @file main.cpp
 * @brief Main entry point for the CUDA test application.
 * @author Vishank Singh, https://github.com/VishankSingh
 */
#include "test.h"
#include "hello.cuh"
#include "hello.h"

#include "utils/compute.h"
#include "tensors/tensors.h"

#include <iostream>

#include <cuda_runtime.h>

__global__ void dummy() {}

int main(int argc, char *argv[]) {
    std::cout << "\033[1m\033[32m[Compute Type] " << "\033[0m"
              << lib::compute::toString(lib::compute::computeType)
              << std::endl;
    lib::compute::setComputeType(lib::compute::ComputeType::CUDA);

    std::cout << "\033[1m\033[32m[Compute Type] " << "\033[0m"
              << lib::compute::toString(lib::compute::computeType)
              << std::endl;
    
    std::cout << std::endl;

    // printTest();
    // launchHelloWorldKernel();

    // helloWorld(1);

    // Test Tensor functionality
    using namespace lib::tensors;
    using namespace lib::compute;

    Tensor<float> tensor({2, 3}, ComputeType::CPU);

    std::cout << tensor.info() << std::endl;
    tensor.allocateData();
    tensor = Tensor<float>::eye(3, ComputeType::CPU);
    std::cout << tensor.info() << std::endl;

    tensor.print();

    std::cout << std::endl;

    Tensor<float> tensor2({2, 3}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 
        ComputeType::CPU);
    std::cout << tensor2.info() << std::endl;



    // tensor2.moveToCompute();
    tensor2.moveToCpu();
    
    tensor2 = Tensor<float>({2, 3}, 
        {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, 
        ComputeType::CPU);
    std::cout << tensor2.info() << std::endl;
    tensor2.print();

    



    // std::cout << std::endl;


    std::cout << tensor2.at({0, 1}) << std::endl;
    tensor2.at({0, 1}) = 10.0f;
    std::cout << tensor2.at({0, 1}) << std::endl;
    tensor2.print();



    // factory methods 
    
    Tensor<float> zeros_tensor = Tensor<float>::zeros({2, 3}, ComputeType::CPU);
    std::cout << zeros_tensor.info() << std::endl;
    zeros_tensor.print();

    std::cout << std::endl;

    Tensor<float> ones_tensor = Tensor<float>::ones({2, 3}, ComputeType::CPU);
    std::cout << ones_tensor.info() << std::endl;
    ones_tensor.print();

    std::cout << std::endl;

    Tensor<float> full_tensor = Tensor<float>::full({2, 3}, 5.0f, ComputeType::CPU);
    std::cout << full_tensor.info() << std::endl;
    full_tensor.print();

    std::cout << std::endl;

    Tensor<float> arange_tensor = Tensor<float>::arange(0.0f, 10.0f, 0.9f, ComputeType::CPU);
    std::cout << arange_tensor.info() << std::endl;
    arange_tensor.print();

    std::cout << std::endl;

    Tensor<float> eye_tensor = Tensor<float>::eye(4, ComputeType::CPU);
    std::cout << eye_tensor.info() << std::endl;
    eye_tensor.print();

    std::cout << std::endl;

    Tensor<float> random_tensor = Tensor<float>::random({2, 3}, 0.0f, 1.0f, ComputeType::CPU);
    std::cout << random_tensor.info() << std::endl;
    random_tensor.print();
    // std::cout << std::endl;


    Tensor<float> tensor3({2}, {1.0f, 2.0f}, ComputeType::CPU);
    Tensor<float> tensor4({1}, ComputeType::CPU);
    tensor4.print();
    std::cout << tensor4.info() << std::endl;
    tensor4 = tensor3;
    std::cout << tensor4.info() << std::endl;
    tensor4.print();



    tensor3.print();
    std::cout << tensor3.info() << std::endl;

    Tensor<float> tensor5({2, 3}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 
        ComputeType::CPU);

    tensor5.print();
    tensor5.moveToCompute();
    
    tensor5 = Tensor<float>();
    std::cout << tensor5.info() << std::endl;
    tensor5.print();
    

    // Tensor<float> tensor6({2,3,3, 2},
    //     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 
    //      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
    //      13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
    //         19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
    //         25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
    //         31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f

         
    //      },
    //     ComputeType::CPU
    // );

    // std::cout << tensor6.info() << std::endl;
    // tensor6.print();

    std::cout << std::endl;




    return 0;
}