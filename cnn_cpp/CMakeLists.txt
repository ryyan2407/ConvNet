cmake_minimum_required(VERSION 3.10)
project(cnn_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

configure_file(include/project_config.hpp.in generated/project_config.hpp @ONLY)

include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/generated)

find_package(OpenMP)
option(CNN_CPP_ENABLE_NATIVE_OPT "Enable architecture-specific optimization flags" ON)

set(CNN_SOURCES
    src/tensor.cpp
    src/utils.cpp
    src/conv2d.cpp
    src/relu.cpp
    src/maxpool2d.cpp
    src/flatten.cpp
    src/linear.cpp
    src/softmax.cpp
    src/sequential.cpp
    src/weights_loader.cpp
    src/model_config.cpp
    src/model_io.cpp
    src/checkpoint_io.cpp
    src/eval_utils.cpp
    src/image_loader.cpp
    src/cross_entropy.cpp
    src/sgd.cpp
)

add_library(cnn_core ${CNN_SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(cnn_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(cnn_core PUBLIC CNN_CPP_USE_OPENMP=1)
endif()
target_compile_options(cnn_core PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
)
if(CNN_CPP_ENABLE_NATIVE_OPT)
    target_compile_options(cnn_core PRIVATE
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU>>:-march=native>
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:Clang>>:-march=native>
    )
endif()

add_executable(cnn_infer src/main.cpp)
target_link_libraries(cnn_infer PRIVATE cnn_core)

add_executable(cnn_first_milestone src/first_milestone.cpp)
target_link_libraries(cnn_first_milestone PRIVATE cnn_core)

add_executable(cnn_train_demo src/train_demo.cpp)
target_link_libraries(cnn_train_demo PRIVATE cnn_core)

add_executable(cnn_mnist_train src/mnist_train.cpp)
target_link_libraries(cnn_mnist_train PRIVATE cnn_core)

add_executable(cnn_eval src/eval.cpp)
target_link_libraries(cnn_eval PRIVATE cnn_core)

add_executable(cnn_benchmark src/benchmark_mnist.cpp)
target_link_libraries(cnn_benchmark PRIVATE cnn_core)

enable_testing()

function(add_cnn_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE cnn_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_cnn_test(test_tensor)
add_cnn_test(test_relu)
add_cnn_test(test_conv2d)
add_cnn_test(test_maxpool2d)
add_cnn_test(test_flatten)
add_cnn_test(test_linear)
add_cnn_test(test_softmax)
add_cnn_test(test_sequential)
add_cnn_test(test_io)
add_cnn_test(test_stage2)
add_cnn_test(test_gradient_check)
add_cnn_test(test_model_io)
add_cnn_test(test_model_config)
add_cnn_test(test_checkpoint_io)
