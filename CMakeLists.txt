cmake_minimum_required(VERSION 3.21)
project(autodiffeng LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

set(SRC_DIR "src")
set(INCLUDE_DIR "include")
set(CUDA_KERNELS_DIR "cuda_kernels")
set(CPU_KERNELS_DIR "cpu_kernels")
set(TEST_DIR "test")

file(GLOB_RECURSE SRC_FILES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE CUDA_KERNELS_FILES "${CUDA_KERNELS_DIR}/*.cu")
file(GLOB_RECURSE CPU_KERNELS_FILES "${CPU_KERNELS_DIR}/*.cpp")
file(GLOB_RECURSE TEST_FILES "${TEST_DIR}/*.cpp")

add_executable(${PROJECT_NAME} ${SRC_FILES} ${CUDA_KERNELS_FILES} ${CPU_KERNELS_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDA_KERNELS_DIR}
    ${CPU_KERNELS_DIR}

    )
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -g -O3)
target_link_libraries(${PROJECT_NAME} PRIVATE m ${CUDAToolkit_LIBRARIES})

if(ENABLE_ASAN)
    message(STATUS "ASAN enabled: Adding AddressSanitizer flags to main target")
    target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    set(ASAN_ENV "ASAN_OPTIONS=detect_leaks=1:leak_check_at_exit=1")
    add_custom_target(run_asan
        COMMAND ${CMAKE_COMMAND} -E env ${ASAN_ENV} ./${PROJECT_NAME}
        DEPENDS ${PROJECT_NAME}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running ${PROJECT_NAME} with ASAN"
    )
endif()


# tests

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
list(REMOVE_ITEM SRC_FILES "${CMAKE_SOURCE_DIR}/src/main.cpp")

add_executable(tests ${SRC_FILES} ${TEST_FILES} ${CUDA_KERNELS_FILES} ${CPU_KERNELS_FILES})
target_include_directories(tests PRIVATE 
    ${INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDA_KERNELS_DIR}
    ${CPU_KERNELS_DIR}
)
target_link_libraries(tests GTest::GTest GTest::Main pthread ${CUDAToolkit_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(tests)

add_custom_target(run
    COMMAND ${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# add_custom_target(debug
#     COMMAND gdb ${PROJECT_NAME}
#     DEPENDS ${PROJECT_NAME}
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
# )

add_custom_target(valgrind
    COMMAND valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(test_run
    COMMAND ./tests
    DEPENDS tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating documentation with Doxygen"
    )
endif()

message(STATUS "Done")
