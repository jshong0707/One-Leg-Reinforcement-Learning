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
    binding/Trajectory.cc
    binding/FSM.cc
    
)

target_include_directories(ctrlbind
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(ctrlbind PRIVATE Eigen3::Eigen)
# 필요시 최적화
target_compile_options(ctrlbind PRIVATE -O3)


# cmake_minimum_required(VERSION 3.15)
# project(ctrlbind LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(pybind11 CONFIG REQUIRED)
# find_package(Eigen3 REQUIRED)
# find_package(Python3 COMPONENTS Interpreter REQUIRED)   # ★ 추가: 파이썬 해석기

# # === 파이썬 mujoco 패키지에서 include/lib 경로 자동 감지 (최소 변경) ===
# execute_process(
#   COMMAND ${Python3_EXECUTABLE} -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
#   OUTPUT_VARIABLE MUJOCO_PY_DIR
#   OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# message(STATUS "MuJoCo (python pkg) dir: ${MUJOCO_PY_DIR}")

# # 파이썬 확장 모듈 빌드
# pybind11_add_module(ctrlbind
#     binding/bindings.cc
#     binding/Controller.cc
#     binding/Trajectory.cc       # ★ 주석 해제
#     binding/FSM.cc              # ★ 주석 해제
# )

# target_include_directories(ctrlbind
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../include
#     ${MUJOCO_PY_DIR}/include    # ★ 추가: mujoco/mujoco.h 찾도록
# )

# # ★ 추가: libmujoco.so 위치
# target_link_directories(ctrlbind PRIVATE ${MUJOCO_PY_DIR}/lib)

# # ★ 추가: MuJoCo 라이브러리 링크
# target_link_libraries(ctrlbind PRIVATE Eigen3::Eigen mujoco)

# # (리눅스에서 종종 필요)
# if(UNIX AND NOT APPLE)
#   target_link_libraries(ctrlbind PRIVATE dl m pthread)
# endif()

# # 필요시 최적화
# target_compile_options(ctrlbind PRIVATE -O3)