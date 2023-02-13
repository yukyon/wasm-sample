cmake_minimum_required(VERSION 3.5)
project(linear)

set(CMAKE_EXECUTABLE_SUFFIX ".html")
set(CMAKE_CXX_STANDARD 17)
set(INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(EMSDK_FLAGS
        "-s ALLOW_MEMORY_GROWTH=1")

if(TFLITE_WITH_XNNPACK_SIMD)
    ADD_DEFINITIONS(-DUSE_XNNPACK_SIMD)
    SET(EMSDK_FLAGS "${EMSDK_FLASGS} -pthread -s USE_PTHREADS -s PTHREAD_POOL_SIZE=4")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EMSDK_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EMSDK_FLAGS}")

add_executable(linear
    ${SRC_DIR}/linear.cc
)

target_include_directories(linear PUBLIC ${INC_DIR})
target_link_libraries(linear tflite)
set_target_properties(linear PROPERTIES LINK_FLAGS "-o linear.html --preload-file model")

file(INSTALL model DESTINATION .)
