cmake_minimum_required(VERSION 3.18)

project(
  cuda-template
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-DSCTL_QUAD_T=__float128)

find_package(pvfmm REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)

add_executable(all_pairs src/all_pairs.cpp src/kernels.cu src/main.cpp)
set_property(TARGET all_pairs PROPERTY CUDA_ARCHITECTURES "70;75;80")
target_include_directories(all_pairs PUBLIC ${PVFMM_INCLUDE_DIR}/pvfmm ${PVFMM_DEP_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(all_pairs ${PVFMM_LIB_DIR}/${PVFMM_STATIC_LIB} MPI::MPI_CXX OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})

add_executable(tc src/tc.cu)
set_property(TARGET tc PROPERTY CUDA_ARCHITECTURES "80")
