cmake_minimum_required(VERSION 3.5)
project(catdog)

set(CMAKE_EXECUTABLE_SUFFIX ".html")
set(CMAKE_CXX_STANDARD 17)
set(INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(EMSDK_FLAGS
        " -pthread -s USE_PTHREADS -s PTHREAD_POOL_SIZE=4")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EMSDK_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EMSDK_FLAGS}")

if(TFLITE_WITH_XNNPACK_SIMD)
    ADD_DEFINITIONS(-DUSE_XNNPACK_SIMD)
endif()

add_executable(catdog
    ${SRC_DIR}/catdog.cc
)

target_include_directories(catdog PUBLIC ${INC_DIR})
target_link_libraries(catdog tflite)
set_target_properties(catdog PROPERTIES LINK_FLAGS "-o catdog.html --preload-file model")

file(INSTALL model DESTINATION .)
