/**
 * @file tensors.cpp
 * @brief 
 * @author Vishank Singh, https://github.com/VishankSingh
 */

#include "tensors/tensors.h"
#include "utils/compute.h"
#include "utils/utils.h"

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <string>
#include <stdexcept>
#include <iostream>
#include <ostream>
#include <sstream>
#include <functional>
#include <typeinfo>



namespace lib::tensors {

template<typename Type__>
void Tensor<Type__>::allocateData() {
    if (is_cpu_allocated_ && data_ != nullptr) {
        deallocateData();
    }
    data_ = new Type__[total_size_];
    is_cpu_allocated_ = true;
}

template<typename Type__>
void Tensor<Type__>::deallocateData() {
    if (!is_cpu_allocated_) return;
    if (is_cpu_allocated_ && data_ != nullptr) {
        delete[] data_;
        data_ = nullptr;
        is_cpu_allocated_ = false;
    }
}

template<typename Type__>
void Tensor<Type__>::allocateComputeData() {
    if (is_compute_allocated_) {
        return;
    }
    if (compute_type_ == compute::ComputeType::CUDA) {
        cudaError_t err = cudaMalloc(&compute_data_, total_size_ * sizeof(Type__));
        if (err != cudaSuccess) {
            compute_data_ = nullptr;
            throw std::runtime_error("CUDA allocation failed: " + std::string(cudaGetErrorString(err)) + " (Error code: " + std::to_string(err) + ")");
        }
        is_compute_allocated_ = true;
    }
}

template<typename Type__>
void Tensor<Type__>::deallocateComputeData() {
    if (!is_compute_allocated_) return;
    if (compute_type_ == compute::ComputeType::CUDA) {
        cudaError_t err = cudaFree(compute_data_);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA deallocation failed: " + std::string(cudaGetErrorString(err)) + " (Error code: " + std::to_string(err) + ")");
        }
    }
    compute_data_ = nullptr;
    is_compute_allocated_ = false;
}

template<typename Type__>
void Tensor<Type__>::copyDataFrom(const Tensor& other) {
    if (shape_ != other.shape_) {
        std::ostringstream oss;
        oss << "Cannot copy data from tensor with different shape. "
            << "Source shape: [";
        for (size_t i = 0; i < other.shape_.size(); ++i) {
            oss << other.shape_[i];
            if (i != other.shape_.size() - 1) oss << ", ";
        }
        oss << "], Destination shape: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            oss << shape_[i];
            if (i != shape_.size() - 1) oss << ", ";
        }
        oss << "]";
        throw std::runtime_error(oss.str());
    }
    
    if (total_size_ != other.total_size_) {
        throw std::runtime_error("Cannot copy data from tensor with different size. Source size: " + std::to_string(other.total_size_) + ", Destination size: " + std::to_string(total_size_));
    } 
    
    if (!is_cpu_allocated_) {
        allocateData();
    }

    if (other.data_ == nullptr) {
        std::cerr << YELLOW_BOLD "[Warning] Copying from an empty tensor" RESET << std::endl;
        std::cerr << YELLOW_BOLD "[Note] This may lead to undefined behavior if the tensor is used after copying." RESET << std::endl;
        // return;
    }

    if (!other.is_cpu_stale_) {
        if (other.data_ == nullptr) {
            data_ = nullptr; 
            return;
        }
        std::copy(other.data_, other.data_ + total_size_, data_);
    } else if (!other.is_compute_stale_) {
        if (other.compute_type_ == lib::compute::ComputeType::CUDA) {
            cudaError_t err = cudaMemcpy(data_, other.compute_data_, total_size_ * sizeof(Type__), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)) + " (Error code: " + std::to_string(err) + ")");
            }
        } else {
            throw std::runtime_error("Unsupported compute type for copy.");
        }
    } else if (other.is_cpu_stale_ && other.is_compute_stale_) {
        throw std::runtime_error("Both CPU and compute copies are stale in source tensor. No valid data to copy.");
    }

    is_cpu_stale_ = false;
    if (is_compute_allocated_) is_compute_stale_ = true;


}

template<typename Type__>
void Tensor<Type__>::moveDataFrom(Tensor&& other) {
    // if (total_size_ != other.total_size_) {
    //     throw std::runtime_error("Cannot move data from tensor with different size. Source size: " + std::to_string(other.total_size_) + ", Destination size: " + std::to_string(total_size_));
    // }

    // if (other.data_ == nullptr) {
    //     std::cerr << YELLOW_BOLD "[Warning] Moving from an empty tensor" RESET << std::endl;
    //     std::cerr << YELLOW_BOLD "[Note] This may lead to undefined behavior if the tensor is used after moving." RESET << std::endl;
    // }

    deallocateData();
    deallocateComputeData();

    if (!other.is_cpu_stale_ && other.data_ != nullptr) {
        data_ = other.data_;
        other.data_ = nullptr;
    } else if (!other.is_compute_stale_ && other.compute_data_ != nullptr) {
        if (other.compute_type_ == lib::compute::ComputeType::CUDA) {
            cudaError_t err = cudaMemcpy(data_, other.compute_data_, total_size_ * sizeof(Type__), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)) + " (Error code: " + std::to_string(err) + ")");
            }
        } else {
            throw std::runtime_error("Unsupported compute type for move.");
        }
    } else if (other.is_cpu_stale_ && other.is_compute_stale_) {
        throw std::runtime_error("Both CPU and compute copies are stale in source tensor. No valid data to move.");
    }



    is_cpu_allocated_ = true;
    is_cpu_stale_ = false;
    is_compute_allocated_ = false;
    is_compute_stale_ = true;
    compute_data_ = nullptr;
    other.is_cpu_allocated_ = false;
    other.is_cpu_stale_ = false;
}

template<typename Type__>
void Tensor<Type__>::dataCpuToCompute() {
    if (!is_cpu_allocated_) {
        std::cerr << YELLOW_BOLD " [Warning] CPU data is not allocated, cannot copy to compute." RESET << std::endl;
        return;
    }

    if (compute_type_ == compute::ComputeType::CPU) {
        std::cerr << YELLOW_BOLD "[Warning] Cannot copy to compute data for CPU compute type." RESET << std::endl;
        return;
    }

    if (!is_compute_allocated_) {
        allocateComputeData();
    }
    if (compute_type_ == compute::ComputeType::CUDA) {
        cudaError_t err = cudaMemcpy(compute_data_, data_, total_size_ * sizeof(Type__), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)) + " (Error code: " + std::to_string(err) + ")");
        }
    } else {
        // std::copy(data_, data_ + total_size_, compute_data_);
        std::cerr << YELLOW_BOLD "[Warning] Copying to compute data for non-CUDA compute type is not implemented." RESET << std::endl;
    }
}

template<typename Type__>
void Tensor<Type__>::dataComputeToCpu() {
    if (!is_compute_allocated_) {
        std::cerr << YELLOW_BOLD " [Warning] Compute data is not allocated, cannot copy to CPU." RESET << std::endl;
        return;
    }
    if (!is_cpu_allocated_) {
        allocateData();
    }
    if (compute_type_ == compute::ComputeType::CUDA) {
        cudaError_t err = cudaMemcpy(data_, compute_data_, total_size_ * sizeof(Type__), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)) + " (Error code: " + std::to_string(err) + ")");
        }
    } else {
        // std::copy(compute_data_, compute_data_ + total_size_, data_);
        std::cerr << YELLOW_BOLD "[Warning] Copying from compute data for non-CUDA compute type is not implemented." RESET << std::endl;
    }
}

template<typename Type__>
size_t Tensor<Type__>::computeOffset(const std::vector<size_t>& indices) const {
    if (indices.size() != strides_.size()) {
        throw std::out_of_range("Indices size (" + std::to_string(indices.size()) + 
                                ") does not match tensor shape size (" + std::to_string(strides_.size()) + ").");
    }
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {

#ifdef DEBUG_INTERNAL
            std::cerr << RED_BOLD "[Error] " RESET;
            std::cerr << "Index " << indices[i] << " out of bounds for dimension " << i
                      << " with size " << shape_[i] << "." << std::endl;
#endif
            throw std::out_of_range("Index " + std::to_string(indices[i]) + " out of bounds for dimension " + std::to_string(i) + 
                                " with size " + std::to_string(shape_[i]) + ".");
        }
        offset += indices[i] * strides_[i];
    }
    return offset;
}

template<typename Type__>
std::vector<size_t> Tensor<Type__>::computeStrides(const std::vector<size_t>& shape) const {
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// Constructors & Destructor ======================================================================
template<typename Type__>
Tensor<Type__>::Tensor(compute::ComputeType compute_type)
    : compute_type_(compute_type) {
    total_size_ = 0;
    is_cpu_allocated_ = false;
    is_cpu_stale_ = false;
    is_compute_allocated_ = false;
    is_compute_stale_ = true;
    data_ = nullptr;
    compute_data_ = nullptr;
    shape_ = {};
    strides_ = {};
}

template<typename Type__>
Tensor<Type__>::Tensor(const std::vector<size_t>& shape, compute::ComputeType compute_type) {
    compute_type_ = compute_type;
    shape_ = shape;
    is_cpu_allocated_ = false;
    is_cpu_stale_ = false;
    is_compute_allocated_ = false;
    is_compute_stale_ = true;
    data_ = nullptr;
    compute_data_ = nullptr;
    
    total_size_ = 1;
    if (shape.empty()) throw std::runtime_error("Shape cannot be empty.");
    for (size_t dim : shape) {
        total_size_ *= dim;
    }
    if (total_size_ == 0) throw std::runtime_error("Total size cannot be zero.");
    strides_ = computeStrides(shape);

    // allocateData();
    // allocateComputeData();
};

template<typename Type__>
Tensor<Type__>::Tensor(const std::vector<size_t>& shape, 
                       const std::initializer_list<Type__>& data,
                       compute::ComputeType compute_type) {
    compute_type_ = compute_type;
    total_size_ = 1;
    for (size_t dim : shape) {
        total_size_ *= dim;
    }
    shape_ = shape;

    if (data.size() != total_size_) throw std::runtime_error("Initializer list size does not match tensor shape.");
    if (total_size_ == 0) throw std::runtime_error("Total size cannot be zero.");
    
    strides_ = computeStrides(shape);
    data_ = nullptr;
    is_cpu_allocated_ = false;
    allocateData();
    
    std::copy(data.begin(), data.end(), data_);
    is_cpu_stale_ = false;

    is_compute_allocated_ = false;
    compute_data_ = nullptr;
    is_compute_stale_ = true; 
}

template<typename Type__>
Tensor<Type__>::Tensor(const Tensor& other, compute::ComputeType compute_type)
    : compute_type_(compute_type), shape_(other.shape_), strides_(other.strides_) {
    total_size_ = other.total_size_;
    deallocateData();
    allocateData();
    copyDataFrom(other);
    is_cpu_stale_ = false;  

    is_compute_allocated_ = false;
    compute_data_ = nullptr;
    is_compute_stale_ = true;
}

template<typename Type__>
Tensor<Type__>::Tensor(Tensor&& other, compute::ComputeType compute_type) {
    compute_type_ = compute_type;
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    is_cpu_allocated_ = false;
    is_compute_allocated_ = false;
    is_cpu_stale_ = false;
    is_compute_stale_ = true;
    data_ = nullptr;
    compute_data_ = nullptr;

    total_size_ = other.total_size_;
    moveDataFrom(std::move(other));

    other.data_ = nullptr;
    other.compute_data_ = nullptr;
    other.is_cpu_allocated_ = false;
    other.is_compute_allocated_ = false;
    other.is_cpu_stale_ = false;
    other.is_compute_stale_ = false;
}

template<typename Type__>
Tensor<Type__>::~Tensor() {
    deallocateData();
    deallocateComputeData();
}
// ================================================================================================

// TODO: check these two
template<typename Type__>
Tensor<Type__>& Tensor<Type__>::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        strides_ = other.strides_;
        total_size_ = other.total_size_;
        compute_type_ = other.compute_type_;
        is_cpu_stale_ = other.is_cpu_stale_;
        is_compute_stale_ = other.is_compute_stale_;

        if (!is_cpu_allocated_ || data_ == nullptr || total_size_ != other.total_size_) {
            deallocateData();
            allocateData();
        }
        copyDataFrom(other);  // Copies data_ and updates stale flags
    }
    return *this;
}

template<typename Type__>
Tensor<Type__>& Tensor<Type__>::operator=(Tensor&& other) {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        total_size_ = other.total_size_;
        compute_type_ = other.compute_type_;
        is_cpu_stale_ = other.is_cpu_stale_;

        moveDataFrom(std::move(other));
    }
    return *this;
}


template<typename Type__>
size_t Tensor<Type__>::ndim() const {
    return shape_.size();
}

template<typename Type__>
size_t Tensor<Type__>::size() const {
    return total_size_;
}

template<typename Type__>
const std::vector<size_t>& Tensor<Type__>::shape() const {
    return shape_;
}

template<typename Type__>
std::string Tensor<Type__>::info() const {
    std::string shape_str = "(";
    for (size_t i = 0; i < shape_.size(); ++i) {
        shape_str += std::to_string(shape_[i]);
        if (i != shape_.size() - 1) shape_str += ", ";
    }
    shape_str += ")";
    return "Tensor(shape=" + shape_str + ", dtype=" + typeid(Type__).name() +", compute=" + compute::toString(compute_type_) + ")";
}

template<typename Type__>
void Tensor<Type__>::print() const {
    // std::vector<Type__> host_data(total_size_);
    // cudaMemcpy(host_data.data(), data_, total_size_ * sizeof(Type__), cudaMemcpyDeviceToHost);

    if (!is_cpu_allocated_ || data_ == nullptr) {
        std::cerr << RED_BOLD "[Error] " 
                  << "Cannot print: data is not allocated." RESET << std::endl;
        return;
    }
    if (shape_.empty()) {
        std::cout << "[]" << std::endl; // Empty tensor
        return;
    }
    std::function<void(size_t, size_t)> print_recursive;
    print_recursive = [&](size_t offset, size_t dim) {
        if (dim == shape_.size() - 1) {
            std::cout << "[";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                std::cout << data_[offset + i * strides_[dim]];
                if (i + 1 < shape_[dim]) std::cout << ", ";
            }
            std::cout << "]";
        } else {
            std::cout << "[\n";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                print_recursive(offset + i * strides_[dim], dim + 1);
                if (i + 1 < shape_[dim]) std::cout << ",\n";
            }
            std::cout << "\n]";
        }
    };

    print_recursive(0, 0);
    std::cout << "\n";
}

// TODO: modify for compute tensors
template<typename Type__>
Type__ Tensor<Type__>::at(const std::vector<size_t>& indices) const {
    size_t offset = computeOffset(indices);
    if (offset >= total_size_) {
        throw std::out_of_range("Index out of bounds: " + std::to_string(offset) + " for tensor size " + std::to_string(total_size_));
    }
    return data_[offset];
}

// TODO: modify for compute tensors
// TODO: when modifying the data, make sure the non-stale data is modified
template<typename Type__>
Type__& Tensor<Type__>::at(const std::vector<size_t>& indices) {
    size_t offset = computeOffset(indices);
    if (offset >= total_size_) {
        throw std::out_of_range("Index out of bounds: " + std::to_string(offset) + " for tensor size " + std::to_string(total_size_));
    }
    return data_[offset];
}

template<typename Type__>
bool Tensor<Type__>::changeComputeType(compute::ComputeType new_compute_type) {
    if (compute_type_ == new_compute_type) {
        return false; 
    }

    if (is_cpu_stale_) {
        dataComputeToCpu(); 
    } else if (!is_cpu_stale_) {
    } 

    if (is_compute_allocated_) {
        deallocateComputeData();
    }

    compute_type_ = new_compute_type;
    is_compute_stale_ = true; 
    return true;
}

template<typename Type__>
void Tensor<Type__>::moveToCompute(bool force) {
    if (is_compute_allocated_) {
        if (!is_compute_stale_ && is_cpu_stale_) {
            std::cerr << YELLOW_BOLD "[Warning] Compute data is already up-to-date." RESET << std::endl;
            if (force) {
                std::cerr << YELLOW_BOLD "[Note] Forcing data to compute." RESET << std::endl;
                dataCpuToCompute();
                is_cpu_stale_ = false; 
                is_compute_stale_ = false;
            } else {
                return; 
            }
        }
    } else {
        dataCpuToCompute(); 
        is_cpu_stale_ = false; 
        is_compute_stale_ = false;
        return;
    }
}

template<typename Type__>
void Tensor<Type__>::moveToCpu(bool force) {
    if (is_cpu_allocated_) {
        if (!is_cpu_stale_ && is_compute_stale_) {
            std::cerr << YELLOW_BOLD "[Warning] CPU data is already up-to-date." RESET << std::endl;
            if (force) {
                std::cerr << YELLOW_BOLD "[Note] Forcing data to CPU." RESET << std::endl;
                dataComputeToCpu();
                is_compute_stale_ = false; 
                is_cpu_stale_ = false;
            } else {
                return; 
            }
        }
    } else {
        dataComputeToCpu(); 
        is_compute_stale_ = false; 
        is_cpu_stale_ = false;
        return;
    }
}

template<typename Type__>
Tensor<Type__> Tensor<Type__>::zeros(const std::vector<size_t>& shape, compute::ComputeType compute_type) {
    Tensor<Type__> tensor(shape, compute_type);
    tensor.allocateData();
    std::fill(tensor.data_, tensor.data_ + tensor.total_size_, Type__(0));
    tensor.is_cpu_stale_ = false;  
    tensor.is_compute_stale_ = true;
    return tensor;
}

template<typename Type__>
Tensor<Type__> Tensor<Type__>::ones(const std::vector<size_t>& shape, compute::ComputeType compute_type) {
    Tensor<Type__> tensor(shape, compute_type);
    tensor.allocateData();
    std::fill(tensor.data_, tensor.data_ + tensor.total_size_, Type__(1));
    tensor.is_cpu_stale_ = false;  
    tensor.is_compute_stale_ = true;
    return tensor;
}

template<typename Type__>
Tensor<Type__> Tensor<Type__>::full(const std::vector<size_t>& shape, Type__ value, compute::ComputeType compute_type) {
    Tensor<Type__> tensor(shape, compute_type);
    tensor.allocateData();
    std::fill(tensor.data_, tensor.data_ + tensor.total_size_, value);
    tensor.is_cpu_stale_ = false;  
    tensor.is_compute_stale_ = true;
    return tensor;
}

template<typename Type__>
Tensor<Type__> Tensor<Type__>::arange(Type__ start, Type__ end, Type__ step, compute::ComputeType compute_type) {
    if (step == 0) {
        throw std::invalid_argument("Step cannot be zero.");
    }
    if ((end - start) * step < 0) {
        throw std::invalid_argument("Invalid range for arange with given step.");
    }

    size_t size = static_cast<size_t>((end - start) / step);
    std::vector<size_t> shape = {size};
    Tensor<Type__> tensor(shape, compute_type);
    tensor.allocateData();
    
    for (size_t i = 0; i < size; ++i) {
        tensor.data_[i] = start + i * step;
    }
    
    tensor.is_cpu_stale_ = false;  
    tensor.is_compute_stale_ = true;
    return tensor;
}

template<typename Type__>
Tensor<Type__> Tensor<Type__>::eye(size_t n, compute::ComputeType compute_type) {
    std::vector<size_t> shape = {n, n};
    Tensor<Type__> tensor(shape, compute_type);
    tensor.allocateData();
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            tensor.data_[i * n + j] = (i == j) ? Type__(1) : Type__(0);
        }
    }
    
    tensor.is_cpu_stale_ = false;  
    tensor.is_compute_stale_ = true;
    return tensor;
}

template<typename Type__>
Tensor<Type__> Tensor<Type__>::random(const std::vector<size_t>& shape, Type__ min, Type__ max, compute::ComputeType compute_type) {
    if (min >= max) {
        throw std::invalid_argument("Invalid range for random tensor: min should be less than max.");
    }
    Tensor<Type__> tensor(shape, compute_type);

    tensor.allocateData();
    
    for (size_t i = 0; i < tensor.total_size_; ++i) {
        tensor.data_[i] = static_cast<Type__>(min + static_cast<Type__>(rand()) / (static_cast<Type__>(RAND_MAX / (max - min))));
    }
    
    tensor.is_cpu_stale_ = false;  
    tensor.is_compute_stale_ = true;
    return tensor;
}


}


// Explicit template instantiation for common types
template class lib::tensors::Tensor<float>;
template class lib::tensors::Tensor<double>;
template class lib::tensors::Tensor<int>;

