/**
 * @file utils.h
 * @brief 
 * @author Vishank Singh, https://github.com/VishankSingh
 */
#pragma once

#define RED_BOLD "\033[1m\033[31m"
#define GREEN_BOLD "\033[1m\033[32m"
#define YELLOW_BOLD "\033[1m\033[33m"
#define RESET "\033[0m"

// #define DEBUG_INTERNAL

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << RED_BOLD "[Error] " RESET << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }
