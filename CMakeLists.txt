cmake_minimum_required(VERSION 3.10)
# Set the project name
project(Svd VERSION 1.0)
# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# To locate "custom"/manually added libraries
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Locate libraries and headers (see Files in ./cmake/Modules/)
# find_package(Vitis REQUIRED)
find_package(Vivado REQUIRED)
find_package(OpenCv REQUIRED)

# set(HLS_INCLUDE_DIRS ${VITIS_INCLUDE_DIRS})
# set(HLS_INCLUDE_DIRS ${VIVADO_INCLUDE_DIRS})
# message(${HLS_INCLUDE_DIRS})

# Add all definitions
if (WIN32)
	add_compile_definitions(IMAGE_OUTPUT_PATH="C:/Users/ste/phd/hls_projects/hls_svd/data")
else()
	add_compile_definitions(IMAGE_OUTPUT_PATH="/mnt/c/Users/ste/phd/hls_projects/hls_svd/data")
endif()
# The following definitions is required for compiling half-precision numbers.
add_compile_definitions(HLS_NO_XIL_FPO_LIB)
# add_compile_definitions(USE_FLOAT)
add_compile_definitions(DEBUG_LEVEL=2)

# Move executable in bin/, along side the DLLs (copied)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
file(COPY ${OpenCv_LIBS} DESTINATION ${EXECUTABLE_OUTPUT_PATH})

# NOTE: an object file becomes a library. All libraries/objects must be LINKED later!
# Tell the application where to find the other CMake config files.
enable_testing()
add_subdirectory(${CMAKE_SOURCE_DIR}/src)

add_executable(Svd ${CMAKE_CURRENT_SOURCE_DIR}/src/svd.cpp)
target_link_libraries(Svd SVD)
set_target_properties(Svd PROPERTIES LINK_FLAGS -Wl,--stack,10485760)
