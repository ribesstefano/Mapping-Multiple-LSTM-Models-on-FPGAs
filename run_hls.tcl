source tcl/utils.tcl
source tcl/lstm_params.tcl

set PRJ_PATH [pwd]

exec mkdir -p -- ./hls_prj
exec mkdir -p -- ./hls_prj/reports
cd hls_prj

set USE_VITIS 1
# ==============================================================================
# Setups
# ==============================================================================
set reset_project 1
set csim 0
set build_only 0
set synth 1
set cosim 0
set export 1
set place_and_route 0
set report_info 1
# ==============================================================================
# HLS Synthesis Options + Platform Selection
# ==============================================================================
set scheduler_effort "high" ;# medium
set relax_ii 0
set use_hlslib 0
set use_zedboard 1
set use_zcu104_pynq 0
set use_zcu102_vassilis 0

if {${use_zedboard}} {
    set board_name "ZedBoard"
} elseif {${use_zcu104_pynq}} {
    set board_name "ZCU104"
} else {
    set board_name "ZCU102"
}
# ==============================================================================
# Hardware parameters
# ==============================================================================

# ==============================================================================
# Top function name, testbench file
# ==============================================================================
# NOTE: The namespace must also be included.
set TOP "HlsSvdKernel" ;# "HlsDenseSvd" ;# "HlsLstmSvd" ;# "HlsLstmSvd" ; #"HlsKernelS" ;# "HlsGemvKernel" ;#"HlsAxisKernelU" ;#"svd::SvdModel2LstmSDSoCV2"
set TB "test_v_kernel" ; #"test_gemv_kernel"
set SRC_DIR "" ;# Or just leave it empty for including all sub-dirs too.
set SRC_LIST [list ""] ;# If empty, it will include all files in SRC_DIR subdirs
# ==============================================================================
# Project name
# ==============================================================================
set prefix ":"
set TOP_NO_NAMESPACE "HlsSvdKernel" ;# "HlsDenseSvd" ;# "HlsLstmSvd" ;# "HlsLstmSvd" ; #"HlsKernelS" ;# "HlsGemvKernel" ; #"HlsAxisKernelU" ;# [ regsub ***=${prefix} ${TOP} "" string ]
puts ${TOP_NO_NAMESPACE}
set PROJECT_NAME "vitis_${board_name}_${TOP_NO_NAMESPACE}"
# ==============================================================================
# Defines
# ==============================================================================
# The HLS_NO_XIL_FPO_LIB flag is used to compile half precision numbers.
set DEFINES "-DHLS_NO_XIL_FPO_LIB"
append DEFINES ""

if {${use_zcu104_pynq}} {
    # Change the AXI port width from default 64bit to 128bit
    append DEFINES " -DAXI_PORT_WIDTH=128"
} else {
    append DEFINES " -DAXI_PORT_WIDTH=64"
}

append_lstm_params DEFINES
# append DEFINES " -DDEBUG_INTERNAL_STREAMS"
# append DEFINES " -DUSE_BLAS"
# ==============================================================================
# Linker Flags
# ==============================================================================
set LDFLAGS ""
# set LDFLAGS "-lpthread -fopenmp"
# append LDFLAGS " /usr/local/lib/libblas.a"
# ==============================================================================
# TB arguments
# ==============================================================================
set ARGV "2 16 16"
# ==============================================================================
# CFlags
# ==============================================================================
# NOTE(21/02/2019): the '-fno-builtin' is suggested by Xilinx when using
# the set_directive_resource option.
if {${USE_VITIS}} {
    set CXXSTD "-std=c++14" ;#"-std=c++14 -fno-builtin" ; #"-std=c++1y"
} else {
    set CXXSTD "-std=c++0x -fno-builtin"
}
if {${cosim}} {
    set CFLAGS "-O3 ${CXXSTD} -I${PRJ_PATH}/include/${SRC_DIR}/ -DCOSIM_DESIGN ${DEFINES} -I/usr/local/include"
} else {
    set CFLAGS "-O3 ${CXXSTD} -I${PRJ_PATH}/include/${SRC_DIR}/ ${DEFINES} -I/usr/local/include"
}
# ==============================================================================
# Open Project and Add Files
# ==============================================================================
if {${reset_project}} {
    open_project -reset ${PROJECT_NAME}
} else {
    open_project ${PROJECT_NAME}
}

set_top ${TOP}

set HLS_REPORT_PATH "${PROJECT_NAME}/solution_${TOP}/syn/report/"
set REPORT_DIR "${PRJ_PATH}/hls_prj/reports"
set REPORT_FILE_PATH "${PRJ_PATH}/hls_prj/reports/"
# set VIVADO_LIB "C:/Xilinx/Vivado/2018.3/include/"
set BLAS_LIB "C:/Users/ste/.caffe/dependencies/libraries_v140_x64_py27_1.1.0/libraries/lib/libopenblas.a"
set BLAS_LIB_DIR "C:/Users/ste/.caffe/dependencies/libraries_v140_x64_py27_1.1.0/libraries/lib"

# Get Source Files (1st argument: regex, 2nd argument: excluded directory)
set src_files [findFiles "${PRJ_PATH}/src/${SRC_DIR}/" "*.cpp" "${PRJ_PATH}/src/testbenches"]
set include_files [findFiles "${PRJ_PATH}/include/${SRC_DIR}/" "*.h" "${PRJ_PATH}/include/testbenches"]

if {${reset_project}} {
    add_files ${PRJ_PATH}/src/kernel/u_kernel.cpp -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/src/kernel/s_kernel.cpp -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/src/kernel/v_kernel.cpp -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/src/kernel/svd_kernel.cpp -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/src/layers/dense/hls/dense_svd.cpp -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/src/layers/lstm/hls/lstm_svd.cpp -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/src/hls_utils/adder_tree.cpp -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/src/hls_utils/hls_metaprogramming.cpp -cflags ${CFLAGS}

    add_files ${PRJ_PATH}/include/kernel/u_kernel.h -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/include/kernel/s_kernel.h -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/include/kernel/v_kernel.h -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/include/kernel/svd_kernel.h -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/include/layers/dense/hls/dense_svd.h -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/include/layers/lstm/hls/lstm_svd.h -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/include/math_utils/activation_functions.h -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/include/hls_utils/adder_tree.h -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/include/hls_utils/hls_metaprogramming.h -cflags ${CFLAGS}
    add_files ${PRJ_PATH}/include/dma/svd_parameters.h -cflags ${CFLAGS}

    # foreach f ${include_files} {
    #     # File svd.h contains main()
    #     if {${f} eq "${PRJ_PATH}/include/svd.h"} {
    #     } else {
    #         add_files ${f} -cflags ${CFLAGS}
    #     }
    # }
    # foreach f ${src_files} {
    #     # File svd.cpp contains main()
    #     if {${f} eq "${PRJ_PATH}/src/svd.cpp"} {
    #     } else {
    #         add_files ${f} -cflags ${CFLAGS}
    #     }
    # }

    # Add Testbench Files
    if {${csim} || ${cosim}} {
        # TB Files (to avoid including multiple files with main() in them)
        add_files -tb ${PRJ_PATH}/src/testbenches/${TB}.cpp -cflags ${CFLAGS}
        add_files -tb ${PRJ_PATH}/include/testbenches/${TB}.h -cflags ${CFLAGS}
    }
}

if {${USE_VITIS}} {
    open_solution -flow_target vivado -reset "solution_${TOP_NO_NAMESPACE}"
} else {
    open_solution "solution_${TOP_NO_NAMESPACE}"
}
# ==============================================================================
# Set Part
# ==============================================================================
if {${reset_project}} {
    if {${use_zedboard}} {
        # ZedBoard
        set_part {xc7z020clg484-1} ;#-tool vivado
    } else {
        if {${use_zcu104_pynq}} {
            # Pynq ZCU104 Board
            set_part {xczu7ev-ffvc1156-2-e}
            # 100 MHz --> 10 ns
            # 200 MHz --> 5 ns
            # 300 MHz --> 3.333 ns
            create_clock -period 5 -name default
        } elseif {${use_zcu102_vassilis}} {
            # Ultrascale+ ZCU102
            set_part {xczu9eg-ffvb1156-2-i} ;#-tool vivado
            create_clock -period 10 -name default
        } else {
            # ZedBoard (default)
            set_part {xc7z020clg484-1} ;#-tool vivado
            create_clock -period 10 -name default
        }
    }
}
# Lab board Artix-7
# set_part {xc7a100tcsg324-1} -tool vivado

# set_directive_top -name svd_module sum_streams_v2_test
# set_top svd_module

# ==============================================================================
# Configure HLS
# ==============================================================================
config_compile -name_max_length=12

if {${relax_ii}} {
    config_schedule -effort ${scheduler_effort} -relax_ii_for_timing=1
} else {
    config_schedule -effort ${scheduler_effort} -relax_ii_for_timing=0
}

# config_sdx -target sdx ;# -optimization_level 3

if {${use_zedboard}} {
    config_interface -m_axi_addr64=0
}
if {${USE_VITIS}} {
    config_interface -m_axi_auto_max_ports=1 -m_axi_offset=slave
}

config_core DSP48 -latency 3
# config_dataflow -default_channel fifo ;#pingpong
set MAX_DEPTH 65536
# config_dataflow -fifo_depth=${MAX_DEPTH} -start_fifo_depth=${MAX_DEPTH} -scalar_fifo_depth=${MAX_DEPTH} -task_level_fifo_depth=${MAX_DEPTH} -override_user_fifo_depth=${MAX_DEPTH}

# ==============================================================================
# Start C-Simulation
# ==============================================================================
if {${csim}} {
    if {${build_only}} {
        csim_design -clean -O -ldflags ${LDFLAGS} -argv ${ARGV} -setup
    } else {
        csim_design -clean -O -ldflags ${LDFLAGS} -argv ${ARGV}
    }
}
# ==============================================================================
# Start Synthesis
# ==============================================================================
if {${synth}} {
    csynth_design

    if {${report_info}} {
        puts "================================================================"
        puts "\[INFO\] Reporting information"
        puts "================================================================"

        set FILENAME "${REPORT_FILE_PATH}/${board_name}_${TOP_NO_NAMESPACE}.rpt"
        set fin [open ${HLS_REPORT_PATH}/${TOP}_csynth.rpt r]
        set fout [open ${FILENAME} a]

        # grep "\\+ Latency \\(clock cycles\\)" 8 $fin $fout
        grep "== Performance Estimates" 18 $fin $fout
        grep "== Utilization Estimates" 20 $fin $fout

        close $fin
        close $fout
    }
}
# ==============================================================================
# Start Cosimulation
# ==============================================================================
if {${cosim}} {

    if {${USE_VITIS}} {
        cosim_design -trace_level port -ldflags ${LDFLAGS} -argv ${ARGV} \
            -enable_dataflow_profiling
            # -enable_fifo_sizing
            # -disable_deadlock_detection
            # -disable_dependency_check
    } else {
        cosim_design -trace_level port -ldflags ${LDFLAGS} -argv ${ARGV} ;#-tool auto -wave_debug
    }


    if {${report_info}} {
        puts "================================================================"
        puts "\[INFO\] Reporting information"
        puts "================================================================"

        set REPORT_FILENAME "${REPORT_FILE_PATH}/${board_name}_${TOP_NO_NAMESPACE}.rpt"
        set HLS_REPORT_PATH "${PROJECT_NAME}/solution_${TOP_NO_NAMESPACE}/sim/report/"
        set fin [open ${HLS_REPORT_PATH}/${TOP}_cosim.rpt r]
        set fout [open ${REPORT_FILENAME} a]

        grep "Simulation tool" 10 $fin $fout

        close $fin
    }
}
# ==============================================================================
# Export Design
# ==============================================================================
# config_rtl -vivado_phy_opt all
if {${export}} {
    if {${place_and_route}} {
        export_design -flow impl -rtl verilog -format ip_catalog
    } else {
        export_design -format ip_catalog
    }
}

puts "================================================================"
puts "\[INFO\] Closing project: ./hls_prj/${PROJECT_NAME}"
puts "================================================================"

exit

cd ${PRJ_PATH}