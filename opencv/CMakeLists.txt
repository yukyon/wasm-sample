project(opencv)

option(ENABLE_SIMD OFF)

add_library(opencv INTERFACE)

set(OPENCV_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(OPENCV_INCLUDE_PATH "${OPENCV_PATH}/include/opencv4")
set(OPENCV_LIB_PATH "${OPENCV_PATH}/lib")

if(ENABLE_SIMD)
    STRING(APPEND OPENCV_LIB_PATH "/simd")
else()
    STRING(APPEND OPENCV_LIB_PATH "/nonsimd")
endif()

set(OPENCV_LIBS
    ${OPENCV_LIB_PATH}/liblibprotobuf.a
    ${OPENCV_LIB_PATH}/libopencv_calib3d.a
    ${OPENCV_LIB_PATH}/libopencv_core.a
    ${OPENCV_LIB_PATH}/libopencv_dnn.a
    ${OPENCV_LIB_PATH}/libopencv_features2d.a
    ${OPENCV_LIB_PATH}/libopencv_flann.a
    ${OPENCV_LIB_PATH}/libopencv_imgproc.a
    ${OPENCV_LIB_PATH}/libopencv_objdetect.a
    ${OPENCV_LIB_PATH}/libopencv_photo.a
    ${OPENCV_LIB_PATH}/libopencv_video.a
    ${OPENCV_LIB_PATH}/libquirc.a
    ${OPENCV_LIB_PATH}/libzlib.a
)

target_link_libraries(opencv INTERFACE ${OPENCV_LIBS})
target_include_directories(opencv INTERFACE ${OPENCV_INCLUDE_PATH})
