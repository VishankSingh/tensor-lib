/**
 * @file compute.h
 * @brief Header file for compute type detection.
 * @author Vishank Singh, https://github.com/VishankSingh
 */
#pragma once

namespace lib::compute {

enum class ComputeType {
    CPU,
    CUDA
};

const char* toString(lib::compute::ComputeType type);

extern ComputeType computeType;
ComputeType detectComputeType();

void setComputeType(ComputeType type);


} // namespace lib::compute
