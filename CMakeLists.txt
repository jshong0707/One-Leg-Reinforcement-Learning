cmake_minimum_required(VERSION 3.15)
project(ctrlbind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

# 파이썬 확장 모듈 빌드
pybind11_add_module(ctrlbind
    binding/bindings.cc
    binding/Controller.cc

)

target_link_libraries(ctrlbind PRIVATE Eigen3::Eigen)
# 필요시 최적화
target_compile_options(ctrlbind PRIVATE -O3)
