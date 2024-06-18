cmake_minimum_required(VERSION 3.0.0)
project(linear-regression VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

set(XTL_SRC_DIR          ${CMAKE_CURRENT_SOURCE_DIR}/libs/xtl)
set(XSIMD_SRC_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/libs/xsimd)
set(XTENSOR_BLAS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/xtensor-blas)
set(XTENSOR_SRC_DIR      ${CMAKE_CURRENT_SOURCE_DIR}/libs/xtensor)
set(RAYLIB_SRC_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/libs/raylib)

set(xtensor_DIR "${CMAKE_SOURCE_DIR}/build/_deps/xtensor-build")
set(xtl_DIR "${CMAKE_SOURCE_DIR}/build/_deps/xtl-build")


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs)

set(MAIN_SRC
${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(linear-regression ${MAIN_SRC})
target_link_libraries(linear-regression PRIVATE raylib lapack blas)
target_include_directories(linear-regression PRIVATE ${XTL_SRC_DIR}/include)
target_include_directories(linear-regression PRIVATE ${XSIMD_SRC_DIR}/include)
target_include_directories(linear-regression PRIVATE ${XTENSOR_BLAS_SRC_DIR}/include)
target_include_directories(linear-regression PRIVATE ${XTENSOR_SRC_DIR}/include)