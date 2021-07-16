cmake_minimum_required(VERSION 3.10)
# Set the project name
project(Svd VERSION 1.0)

# To locate "custom", i.e. manually added, libraries
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Locate external libraries and headers (see Files in ./cmake/Modules/)
# Search for HLS: if Vitis is found, use C++14, else fall back to C++11.
find_package(Vitis)
if(DEFINED VITIS_INCLUDE_DIRS)
	# Specify the C++14 standard
	message("[INFO] Vitis HLS FOUND.")
	set(CMAKE_CXX_STANDARD 14)
	set(CMAKE_CXX_STANDARD_REQUIRED True)
	set(HLS_INCLUDE_DIRS ${VITIS_INCLUDE_DIRS})
	add_compile_definitions(__VITIS_HLS__)
else()
	find_package(Vivado REQUIRED)
	# Specify the C++11 standard
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED True)
	set(HLS_INCLUDE_DIRS ${VIVADO_INCLUDE_DIRS})
endif()
find_package(OpenCv REQUIRED)

message(${HLS_INCLUDE_DIRS})

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

add_compile_definitions(INPUT_SIZE=1024)
add_compile_definitions(HIDDEN_SIZE=512)
add_compile_definitions(NUM_GATES=4)
add_compile_definitions(NUM_SAMPLES=2)
add_compile_definitions(NUM_TILES_U=32)
add_compile_definitions(NUM_ZERO_TILES_U=16)
add_compile_definitions(NUM_TILES_V=64)
add_compile_definitions(NUM_ZERO_TILES_V=8)
add_compile_definitions(NUM_TIMESTEPS=4)

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
