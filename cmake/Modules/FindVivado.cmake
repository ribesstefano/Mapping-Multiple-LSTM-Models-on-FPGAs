# set(VIVADO_INCLUDE_DIRS /mnt/c/Xilinx/Vivado/2018.3/include/)
set(VIVADO_INCLUDE_DIRS C:/Xilinx/Vivado/2018.3/include/)

# NOTE: It handles the REQUIRED, QUIET and version-related arguments of find_package.
# It also sets the <PackageName>_FOUND variable. The package is considered found
# if all variables listed contain valid results, e.g. valid filepaths.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Vivado DEFAULT_MSG VIVADO_INCLUDE_DIRS)
