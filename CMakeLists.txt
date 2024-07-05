cmake_minimum_required(VERSION 3.0.0)
project(linear-regression VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

set(RAYLIB_SRC_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/libs/raylib)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs)

set(MAIN_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(linear-regression ${MAIN_SRC})
target_link_libraries(linear-regression PRIVATE raylib)