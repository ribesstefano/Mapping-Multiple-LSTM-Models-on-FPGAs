
if (WIN32)
	set(OpenCv_DIR C:/Xilinx/Vivado/2018.3/win64/tools/opencv)
	set(OpenCv_INCLUDE_DIRS C:/Xilinx/Vivado/2018.3/win64/tools/opencv/include)
else()
	set(OpenCv_DIR /mnt/c/Xilinx/Vivado/2018.3/win64/tools/opencv)
	set(OpenCv_INCLUDE_DIRS /mnt/c/Xilinx/Vivado/2018.3/win64/tools/opencv/include)
endif()

file(GLOB OpenCv_LIBS ${OpenCv_DIR}/*.dll)

# NOTE: It handles the REQUIRED, QUIET and version-related arguments of find_package.
# It also sets the <PackageName>_FOUND variable. The package is considered found
# if all variables listed contain valid results, e.g. valid filepaths.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCv DEFAULT_MSG OpenCv_INCLUDE_DIRS)