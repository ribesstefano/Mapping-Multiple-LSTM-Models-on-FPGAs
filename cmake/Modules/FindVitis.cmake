if (WIN32)
	set(VITIS_INCLUDE_DIRS D:/Programs/Xilinx/Vitis_HLS/2021.1/include/)
else()
	set(VITIS_INCLUDE_DIRS /mnt/d/Programs/Xilinx/Vitis_HLS/2021.1/include/)
endif()

# NOTE: It handles the REQUIRED, QUIET and version-related arguments of find_package.
# It also sets the <PackageName>_FOUND variable. The package is considered found
# if all variables listed contain valid results, e.g. valid filepaths.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Vitis DEFAULT_MSG VITIS_INCLUDE_DIRS)