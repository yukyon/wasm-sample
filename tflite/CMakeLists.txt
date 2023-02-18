project(tflite)

option(ENABLE_SIMD OFF)

add_library(tflite INTERFACE)

set(TFLITE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(TFLITE_INCLUDE_PATH "${TFLITE_PATH}/include")
set(TFLITE_LIB_PATH "${TFLITE_PATH}/lib")

if(ENABLE_SIMD)
    STRING(APPEND TFLITE_LIB_PATH "/simd")
else()
    STRING(APPEND TFLITE_LIB_PATH "/nonsimd")
endif()

set(TFLITE_LIBS
    ${TFLITE_LIB_PATH}/libclog.a
    ${TFLITE_LIB_PATH}/libcpuinfo.a
    ${TFLITE_LIB_PATH}/libfarmhash.a
    ${TFLITE_LIB_PATH}/libfft2d_fftsg.a
    ${TFLITE_LIB_PATH}/libfft2d_fftsg2d.a
    ${TFLITE_LIB_PATH}/libflatbuffers.a
    ${TFLITE_LIB_PATH}/libpthreadpool.a
    ${TFLITE_LIB_PATH}/libruy_allocator.a
    ${TFLITE_LIB_PATH}/libruy_apply_multiplier.a
    ${TFLITE_LIB_PATH}/libruy_block_map.a
    ${TFLITE_LIB_PATH}/libruy_blocking_counter.a
    ${TFLITE_LIB_PATH}/libruy_context.a
    ${TFLITE_LIB_PATH}/libruy_context_get_ctx.a
    ${TFLITE_LIB_PATH}/libruy_cpuinfo.a
    ${TFLITE_LIB_PATH}/libruy_ctx.a
    ${TFLITE_LIB_PATH}/libruy_denormal.a
    ${TFLITE_LIB_PATH}/libruy_frontend.a
    ${TFLITE_LIB_PATH}/libruy_prepacked_cache.a
    ${TFLITE_LIB_PATH}/libruy_prepare_packed_matrices.a
    ${TFLITE_LIB_PATH}/libruy_system_aligned_alloc.a
    ${TFLITE_LIB_PATH}/libruy_thread_pool.a
    ${TFLITE_LIB_PATH}/libruy_trmul.a
    ${TFLITE_LIB_PATH}/libruy_tune.a
    ${TFLITE_LIB_PATH}/libruy_wait.a
    ${TFLITE_LIB_PATH}/libtensorflow-lite.a
    ${TFLITE_LIB_PATH}/libXNNPACK.a
)

target_link_libraries(tflite INTERFACE ${TFLITE_LIBS})
target_include_directories(tflite INTERFACE ${TFLITE_INCLUDE_PATH})
