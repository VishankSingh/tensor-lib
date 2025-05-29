/**
 * @file tensors.h
 * @brief 
 * @author Vishank Singh, https://github.com/VishankSingh
 */
#pragma once

#include <vector>
#include <cstddef>
#include <string>

#include "utils/compute.h"

// Future: Add Recovery Policy for errors
// TODO: Add support for different compute types (e.g., OpenCL, Metal, etc.)

namespace lib::tensors {

template<typename Type__>
class Tensor {
// private:
public:
    lib::compute::ComputeType compute_type_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t total_size_;
    Type__* data_;
    Type__* compute_data_;
    bool is_cpu_allocated_;
    bool is_compute_allocated_;
    bool is_cpu_stale_;
    bool is_compute_stale_;

    void allocateData();
    void deallocateData();

    void allocateComputeData();
    void deallocateComputeData();
    
    void copyDataFrom(const Tensor& other);
    void moveDataFrom(Tensor&& other);

    void dataCpuToCompute();
    void dataComputeToCpu();

    size_t computeOffset(const std::vector<size_t>& indices) const;
    std::vector<size_t> computeStrides(const std::vector<size_t>& shape) const;
    


public:
    Tensor(compute::ComputeType compute_type = compute::computeType);
    Tensor(const std::vector<size_t>& shape, 
           compute::ComputeType compute_type = compute::computeType);
    Tensor(const std::vector<size_t>& shape, 
           const std::initializer_list<Type__>& data,
           compute::ComputeType compute_type = compute::computeType);
           
    Tensor(const Tensor& other, compute::ComputeType compute_type = compute::computeType);
    Tensor(Tensor&& other, compute::ComputeType compute_type = compute::computeType);
    
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other);

    ~Tensor();

    size_t ndim() const;
    size_t size() const;
    const std::vector<size_t>& shape() const;


    std::string info() const;

    void print() const;

    Type__ at(const std::vector<size_t>& indices) const;
    Type__& at(const std::vector<size_t>& indices);

    bool changeComputeType(compute::ComputeType new_compute_type);

    void moveToCompute(bool force=false);
    void moveToCpu(bool force=false);
    
    // TensorView<Type__> view(const std::vector<size_t>& indices, const std::vector<size_t>& offset) const;

    // some specific initializers
    static Tensor zeros(const std::vector<size_t>& shape, compute::ComputeType compute_type = compute::computeType);
    static Tensor ones(const std::vector<size_t>& shape, compute::ComputeType compute_type = compute::computeType);
    static Tensor full(const std::vector<size_t>& shape, Type__ value, compute::ComputeType compute_type = compute::computeType);
    static Tensor arange(Type__ start, Type__ end, Type__ step = 1, compute::ComputeType compute_type = compute::computeType);
    static Tensor eye(size_t n, compute::ComputeType compute_type = compute::computeType);
    static Tensor random(const std::vector<size_t>& shape, Type__ min = 0, Type__ max = 1, compute::ComputeType compute_type = compute::computeType);


};


// template<typename Type__>
// class TensorView {
// private:
//     Tensor<Type__>* tensor_;
//     std::vector<size_t> shape_;
//     std::vector<size_t> strides_;
//     size_t offset_;

//     size_t computeOffset(const std::vector<size_t>& indices) const;
// public:
//     TensorView(const Tensor<Type__>* tensor,
//                const std::vector<size_t>& shape,
//                const std::vector<size_t>& strides,
//                size_t offset = 0);

    
//     size_t ndim() const;
//     size_t size() const;
//     const std::vector<size_t>& shape() const;

//     Type__ at(const std::vector<size_t>& indices) const;
//     Type__& at(const std::vector<size_t>& indices);

//     void print() const;
// };



} // namespace lib::tensors
