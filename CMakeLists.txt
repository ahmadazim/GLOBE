cmake_minimum_required(VERSION 3.10)
project(GLOBE VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -fopenmp")

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

add_library(GLOBE_LIB STATIC
    src/utils.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(GLOBE_LIB PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(GLOBE_LIB PRIVATE nlopt)

# Executable #1: runPCGsolver
add_executable(runPCGsolver
    src/PCGsolver.cpp
)
target_link_libraries(runPCGsolver PRIVATE GLOBE_LIB)

# Executable #2: runAIREML
add_executable(runAIREML
    src/varCompEst.cpp
)
target_link_libraries(runAIREML PRIVATE GLOBE_LIB)

# Executable #3: runAIREML_reducedModel 
add_executable(runAIREML_reducedModel
    src/varCompEst_reducedModel_noTime.cpp
)
target_link_libraries(runAIREML_reducedModel PRIVATE GLOBE_LIB)

# (Optional) Testing
# enable_testing()
# add_executable(test_solver tests/test_solver.cpp)
# target_link_libraries(test_solver PRIVATE GLOBE_LIB)
# add_test(NAME solver_test COMMAND test_solver)