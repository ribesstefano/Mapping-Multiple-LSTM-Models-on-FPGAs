cmake_minimum_required(VERSION 3.10)

add_library(SVD_DMA STATIC ${CMAKE_SOURCE_DIR}/src/dma/svd_dma.cpp)
target_include_directories(SVD_DMA PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(SVD_DMA PUBLIC ${HLS_INCLUDE_DIRS})
target_compile_options(SVD_DMA PRIVATE -fno-builtin)
target_link_libraries(SVD_DMA SVD_PARAMS)

add_library(WIDTH_CONVERTER STATIC ${CMAKE_SOURCE_DIR}/src/dma/width_converter.cpp)
target_include_directories(WIDTH_CONVERTER PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(WIDTH_CONVERTER PUBLIC ${HLS_INCLUDE_DIRS})
target_compile_options(WIDTH_CONVERTER PRIVATE -fno-builtin)
target_link_libraries(WIDTH_CONVERTER SVD_PARAMS)

add_library(AXIS_LIB STATIC ${CMAKE_SOURCE_DIR}/src/dma/axis_lib.cpp)
target_include_directories(AXIS_LIB PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(AXIS_LIB PUBLIC ${HLS_INCLUDE_DIRS})
target_compile_options(AXIS_LIB PRIVATE -fno-builtin)