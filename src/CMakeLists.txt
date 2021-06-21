cmake_minimum_required(VERSION 3.10)
add_subdirectory(dma)
add_subdirectory(kernel)
add_subdirectory(tb)

# # NOTE: Each library/object will have an identifier and that identifier will then
# # be used to link the final executable, i.e. target_link_libraries(ProjectName LibraryName)
# add_library(AXIS_LIB STATIC ${CMAKE_SOURCE_DIR}/src/axis_lib.cpp)
# target_include_directories(AXIS_LIB PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_include_directories(AXIS_LIB PUBLIC ${VIVADO_INCLUDE_DIRS})

# add_library(PONG STATIC ${CMAKE_SOURCE_DIR}/src/pong.cpp)
# target_include_directories(PONG PUBLIC ${CMAKE_SOURCE_DIR}/include)
# target_include_directories(PONG PUBLIC ${VIVADO_INCLUDE_DIRS})
# target_include_directories(PONG PUBLIC ${OpenCv_INCLUDE_DIRS})
# target_link_libraries(PONG GAME)
# target_link_libraries(PONG DQNET)

add_library(SVD STATIC ${CMAKE_SOURCE_DIR}/src/svd.cpp)
target_include_directories(SVD PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(SVD PUBLIC ${VIVADO_INCLUDE_DIRS})
target_include_directories(SVD PUBLIC ${OpenCv_INCLUDE_DIRS})
target_link_libraries(SVD ${OpenCv_LIBS})
# target_link_libraries(SVD PONG)