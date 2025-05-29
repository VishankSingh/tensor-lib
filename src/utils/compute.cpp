/**
 * @file compute.cpp
 * @brief Implementation of compute type detection.
 * 
 * This file contains the implementation of functions to detect the compute type
 * (CPU or CUDA) based on the availability of CUDA devices.
 *
 * @author Vishank Singh, https://github.com/VishankSingh
 */
#include "utils/compute.h"
#include "utils/utils.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#include <thrust/random.h>


namespace lib::compute {

ComputeType computeType = ComputeType::CPU;

static bool isCudaAvailable() {
    int deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

ComputeType detectComputeType() {
    return computeType;
}

const char* toString(lib::compute::ComputeType type) {
    switch (type) {
        case lib::compute::ComputeType::CPU: return "CPU";
        case lib::compute::ComputeType::CUDA: return "CUDA";
        default: return "Unknown";
    }
}

void setComputeType(ComputeType type) {
    if (type == ComputeType::CPU) {
        if (computeType == ComputeType::CUDA) {
            cudaDeviceReset();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << RED_BOLD 
                          << "[CUDA] Error resetting device: " 
                          << cudaGetErrorString(err) 
                          << RESET << std::endl;
            }
        }
        computeType = ComputeType::CPU;
        std::cout << GREEN_BOLD 
                  << "[Compute Type] CPU selected"
                  << RESET 
                  << std::endl;
    } else if (type == ComputeType::CUDA) {
        if (!isCudaAvailable()) {
            std::cerr << RED_BOLD 
                      << "[CUDA] No CUDA devices available" 
                      << RESET << std::endl;
            std::cerr << RED_BOLD 
                      << "[Compute Type] Falling back to " 
                      << toString(computeType) 
                      << RESET << std::endl;
            return;
        }

        cudaError_t setDeviceErr = cudaSetDevice(0);
        if (setDeviceErr != cudaSuccess) {
            std::cerr << RED_BOLD 
                      << "[CUDA] Error setting device: " 
                      << cudaGetErrorString(setDeviceErr) 
                      << RESET << std::endl;
            std::cerr << RED_BOLD 
                      << "[Compute Type] Falling back to " 
                      << toString(computeType) 
                      << RESET << std::endl;
            return;
        }
        computeType = ComputeType::CUDA;
        std::cout << GREEN_BOLD "[Compute Type] CUDA selected" << RESET << std::endl;

    }
}


} // namespace lib::compute
