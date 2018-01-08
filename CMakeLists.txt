
cmake_minimum_required(VERSION 3.8)
project(nice_admm)

# link_directories(${PROJECT_SOURCE_DIR}/mgcpp/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

set(SOURCE_FILES #${PROJECT_SOURCE_DIR}/main.cpp)
    ${PROJECT_SOURCE_DIR}/blaze.cpp
    ${PROJECT_SOURCE_DIR}/LC_DG_problem.cpp
    ${PROJECT_SOURCE_DIR}/LC_DG_update.cpp
    #${PROJECT_SOURCE_DIR}/LC_DS_update.cpp
    ${PROJECT_SOURCE_DIR}/gradient_projection.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(IPOPT REQUIRED)
find_package(Threads REQUIRED)


target_link_libraries(${PROJECT_NAME}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${IPOPT_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    cblas
    ${CUDA_LIBRARIES})
#    ${LibMGCPP}
#${LibMGCPPBASE})

set(BLAZE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/blaze)
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${BLAZE_INCLUDE_DIRS}
    ${IPOPT_INCLUDE_DIRS}
    #${Boost_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/mgcpp/third_party/boost-outcome/include
    ${PROJECT_SOURCE_DIR}/mgcpp/include)

set(BUILD_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -std=c++17 -Wall -Wextra -DBLAZE_USE_CPP_THREADS")
message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(${PROJECT_NAME}
    PROPERTIES COMPILE_FLAGS "${BUILD_FLAGS}")
