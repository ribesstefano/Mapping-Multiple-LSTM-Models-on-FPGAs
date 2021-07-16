cmake_minimum_required(VERSION 3.10)

add_executable(TEST_U_KERNEL ${CMAKE_SOURCE_DIR}/src/testbenches/test_u_kernel.cpp)
target_include_directories(TEST_U_KERNEL PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(TEST_U_KERNEL PUBLIC ${HLS_INCLUDE_DIRS})
target_include_directories(TEST_U_KERNEL PUBLIC ${OpenCv_INCLUDE_DIRS})
target_link_libraries(TEST_U_KERNEL ${OpenCv_LIBS})
target_link_libraries(TEST_U_KERNEL U_KERNEL)

add_test(NAME TestU_Kernel COMMAND TEST_U_KERNEL)