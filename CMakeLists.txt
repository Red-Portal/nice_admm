
cmake_minimum_required(VERSION 3.8)
project(nice_admm)

# link_directories(${PROJECT_SOURCE_DIR}/mgcpp/lib)

set(SOURCE_FILES ${PROJECT_SOURCE_DIR}/main.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

find_package(CUDA REQUIRED)
find_library(LibMGCPP NAMES mgcpp PATHS ${PROJECT_SOURCE_DIR}/mgcpp/lib)

target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    ${LibMGCPP})

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/mgcpp/third_party/outcome/include
    ${PROJECT_SOURCE_DIR}/mgcpp/include)

set(BUILD_FLAGS
    "${CMAKE_CXX_FLAGS} -g -fPIC -Wall -Wextra")
 message("C++ flags = ${BUILD_FLAGS}")

 set_target_properties(${PROJECT_NAME}
     PROPERTIES COMPILE_FLAGS "${BUILD_FLAGS}")
