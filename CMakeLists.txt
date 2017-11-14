
cmake_minimum_required(VERSION 3.8)
project(nice_admm)

# link_directories(${PROJECT_SOURCE_DIR}/mgcpp/lib)

set(SOURCE_FILES #${PROJECT_SOURCE_DIR}/main.cpp)
    ${PROJECT_SOURCE_DIR}/blaze.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)
find_library(LibMGCPP NAMES mgcpp PATHS ${PROJECT_SOURCE_DIR}/mgcpp/lib)
find_library(LibMGCPPBASE NAMES mgcppbase PATHS ${PROJECT_SOURCE_DIR}/mgcpp/lib)

cuda_add_cublas_to_target(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    ${LibMGCPP}
    ${LibMGCPPBASE})

set(BLAZE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/blaze)
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${BLAZE_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/mgcpp/third_party/boost-outcome/include
    ${PROJECT_SOURCE_DIR}/mgcpp/include)

set(BUILD_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -fPIC -Wall -Wextra -DBLAZE_USE_CPP_THREADS")
message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(${PROJECT_NAME}
    PROPERTIES COMPILE_FLAGS "${BUILD_FLAGS}")
