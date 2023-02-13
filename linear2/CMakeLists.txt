cmake_minimum_required(VERSION 3.5)
project(linear2)

#set(CMAKE_EXECUTABLE_SUFFIX ".html")
set(CMAKE_CXX_STANDARD 17)
set(INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(EMSDK_FLAGS
  "-s MODULARIZE \
   -s ALLOW_MEMORY_GROWTH=1 \
   -s EXPORT_NAME='\"createModule\"' \
   -s EXPORTED_FUNCTIONS='[\"_malloc\", \"_free\"]' \
   -s EXPORTED_RUNTIME_METHODS='[\"ccall\"]'")

if(TFLITE_WITH_XNNPACK_SIMD)
   ADD_DEFINITIONS(-DUSE_XNNPACK_SIMD)
   SET(EMSDK_FLAGS "${EMSDK_FLAGS} -pthread -s USE_PTHREADS -s PTHREAD_POOL_SIZE=4")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EMSDK_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EMSDK_FLAGS}")


add_executable(linear2
    ${SRC_DIR}/linear2.cc
)

target_include_directories(linear2 PUBLIC ${INC_DIR})
target_link_libraries(linear2 tflite)
set_target_properties(linear2 PROPERTIES LINK_FLAGS "--preload-file model")

file(INSTALL model DESTINATION .)
file(INSTALL index.js DESTINATION .)
file(INSTALL wasm.js DESTINATION .)
file(INSTALL index.html DESTINATION .)
