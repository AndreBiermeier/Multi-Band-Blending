# Build and install the C++ graph-cut extension for the Python package.
cmake_minimum_required(VERSION 3.25)
project(multiband_blending LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native
    src/native_src/bindings.cpp
    src/native_src/graphcut/problem_validation.cpp
    src/native_src/graphcut/primal_grid.cpp
    src/native_src/graphcut/dual_grid.cpp
    src/native_src/graphcut/dijkstra.cpp
    src/native_src/graphcut/cut_mask.cpp
    src/native_src/graphcut/dijkstra_mask.cpp
)
target_compile_features(_native PRIVATE cxx_std_20)
target_include_directories(_native PRIVATE src/native_src/include)
if(MSVC)
    target_compile_options(_native PRIVATE /W4 /permissive-)
else()
    target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic)
endif()
install(TARGETS _native LIBRARY DESTINATION multiband_blending
                        RUNTIME DESTINATION multiband_blending)
