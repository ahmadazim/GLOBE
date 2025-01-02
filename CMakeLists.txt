cmake_minimum_required(VERSION 3.10)
project(GLOBE VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NLOPT_ROOT "${CMAKE_SOURCE_DIR}/thirdParty/nlopt")
set(EIGEN_ROOT "${CMAKE_SOURCE_DIR}/thirdParty/eigen")

# Include directories
include_directories(
    "${EIGEN_ROOT}"            # Eigen is header-only
    "${NLOPT_ROOT}/include"    # NLopt headers
    "${PROJECT_SOURCE_DIR}/include"
)

# Link directories
link_directories("${NLOPT_ROOT}/lib64")

# Find/OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP!")
endif()

# set(MKL_ROOT "$ENV{MKLROOT}")
# if(NOT MKL_ROOT)
#     message(WARNING "MKLROOT is not set. Please set MKL_ROOT or MKLROOT environment variable.")
# endif()

# if(EXISTS "${MKL_ROOT}/lib/intel64")
#     link_directories("${MKL_ROOT}/lib/intel64")
# endif()

# set(MKL_LIBS
#     mkl_intel_lp64
#     mkl_gnu_thread
#     mkl_core
#     pthread
#     m
#     dl
# )

# Build a static library from shared code only (no main)
add_library(GLOBE_LIB STATIC
    src/utils.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(GLOBE_LIB PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(GLOBE_LIB PRIVATE
    nlopt
    # ${MKL_LIBS}
)

# Executable #1: runPCGsolver (has main() in PCGsolver.cpp)
add_executable(runPCGsolver
    src/PCGsolver.cpp
)
target_link_libraries(runPCGsolver PRIVATE GLOBE_LIB)

# Executable #2: runAIREML (has main() in runAIREML.cpp)
add_executable(runAIREML
    src/varCompEst.cpp
)
target_link_libraries(runAIREML PRIVATE GLOBE_LIB)

# (Optional) Testing
# enable_testing()
# add_executable(test_solver tests/test_solver.cpp)
# target_link_libraries(test_solver PRIVATE GLOBE_LIB)
# add_test(NAME solver_test COMMAND test_solver)