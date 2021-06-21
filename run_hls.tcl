#
# @brief      Find all files in a directory and return them in a list.
#
# @param      basedir            The directory to start looking in pattern.
# @param      pattern            A pattern, as defined by the glob command, that
#                                the files must match.
# @param      exclude_dirs_list  Ignore searching in specified directories
#
# @return     The list of found files.
#
proc findFiles { basedir pattern exclude_dirs_list } {
    # Fix the directory name, this ensures the directory name is in the
    # native format for the platform and contains a final directory seperator
    set basedir [string trimright [file join [file normalize $basedir] { }]]
    set fileList {}
    # Look in the current directory for matching files, -type {f r}
    # means ony readable normal files are looked at, -nocomplain stops
    # an error being thrown if the returned list is empty
    foreach fileName [glob -nocomplain -type {f r} -path $basedir $pattern] {
        lappend fileList $fileName
    }
    # Now look for any sub direcories in the current directory
    foreach dirName [glob -nocomplain -type {d  r} -path $basedir *] {
        # Recusively call the routine on the sub directory and append any
        # new files to the results
        if {[lsearch -exact ${exclude_dirs_list} $dirName] == -1} {
            set subDirList [findFiles $dirName $pattern $exclude_dirs_list]
            if { [llength $subDirList] > 0 } {
                foreach subDirFile $subDirList {
                    lappend fileList $subDirFile
                }
            }
        }
    }
    return $fileList
}

#
# @brief      Greps a file content and writes matches to a file.
#
# @param      re     Regular expression
# @param      lines  Number of lines to report/include after the found match
# @param      fin    The fin pointer
# @param      fout   The fout pointer
#
proc grep {re lines fin fout} {
    set cnt 0
    set match false
    seek $fin 0
    while {[gets $fin line] >= 0} {
        if [regexp -- $re $line] {
            set cnt 0
            set match true
        }
        if {$match && ($cnt < $lines)} {
            puts $line
            puts $fout $line
            set cnt [expr {$cnt +1}]
        } else {
            set match false
        }
    }
}

set PRJ_PATH [pwd]

exec mkdir -p -- ./hls
exec mkdir -p -- ./hls/reports
cd hls

# ==============================================================================
# Top function name, testbench file
# ==============================================================================
set TOP "hls_pong"
set TB "test_game"
set SRC_DIR "" ;# Or just leave it empty for including all sub-dirs too.
set SRC_LIST [list ""] ;# If empty, it will include all files in SRC_DIR subdirs
# ==============================================================================
# Setups
# ==============================================================================
set reset_project 1
set csim 0
set build_only 0
set synth 1
set cosim 0
set export 0
set place_and_route 0
set report_info 1
# ==============================================================================
# HLS Synthesis Options + Platform Selection
# ==============================================================================
set scheduler_effort "medium"
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
# Project name
# ==============================================================================
set PROJECT_NAME "${board_name}_${TOP}"
# ==============================================================================
# Defines
# ==============================================================================
# The HLS_NO_XIL_FPO_LIB flag is used to compile hlaf precision numbers.
set DEFINES "-DHLS_NO_XIL_FPO_LIB"
append DEFINES ""

if {${use_zcu104_pynq}} {
    # Change the AXI port width from default 64bit to 128bit
    append DEFINES " -DAXI_PORT_WIDTH=128"
} else {
    append DEFINES " -DAXI_PORT_WIDTH=64"
}
# append DEFINES " -DDEBUG_INTERNAL_STREAMS"
# append DEFINES " -DUSE_BLAS"
# ==============================================================================
# Linker Flags
# ==============================================================================
set LDFLAGS "-lpthread -fopenmp"
# append LDFLAGS " /usr/local/lib/libblas.a"
# ==============================================================================
# TB arguments
# ==============================================================================
set ARGV ""
# ==============================================================================
# CFlags
# ==============================================================================
# NOTE(21/02/2019): the '-fno-builtin' is suggested by Xilinx when using
# the set_directive_resource option.
if {${cosim}} {
    set CFLAGS "-O3 -g -std=c++0x -fno-builtin -I${PRJ_PATH}/include/${SRC_DIR}/ -DCOSIM_DESIGN ${DEFINES} -I/usr/local/include"
} else {
    set CFLAGS "-O3 -g -std=c++0x -fno-builtin -I${PRJ_PATH}/include/${SRC_DIR}/ ${DEFINES} -I/usr/local/include"
}
# ==============================================================================
# Open Project and Add Files
# ==============================================================================
if {${reset_project}} {
    open_project -reset hls_${PROJECT_NAME}
} else {
    open_project hls_${PROJECT_NAME}
}

set_top ${TOP}

set HLS_REPORT_PATH "hls_${PROJECT_NAME}/solution_${TOP}/syn/report/"
set REPORT_DIR "${PRJ_PATH}/hls/reports"
set REPORT_FILE_PATH "${PRJ_PATH}/hls/reports/"
set VIVADO_LIB "C:/Xilinx/Vivado/2018.3/include/"
set BLAS_LIB "C:/Users/ste/.caffe/dependencies/libraries_v140_x64_py27_1.1.0/libraries/lib/libopenblas.a"
set BLAS_LIB_DIR "C:/Users/ste/.caffe/dependencies/libraries_v140_x64_py27_1.1.0/libraries/lib"

# Get Source Files (1st argument: regex, 2nd argument: excluded directory)
set src_files [findFiles "${PRJ_PATH}/src/${SRC_DIR}/" "*.cpp" "${PRJ_PATH}/src/tb"]
set include_files [findFiles "${PRJ_PATH}/include/${SRC_DIR}/" "*.h" "${PRJ_PATH}/include/tb"]

if {${reset_project}} {
    foreach f ${src_files} {
        add_files ${f} -cflags ${CFLAGS}
    }
    foreach f ${include_files} {
        add_files ${f} -cflags ${CFLAGS}
    }
    # if {llength $SRC_LIST -eq 0} {
    #     foreach f ${src_files} {
    #         add_files ${f} -cflags ${CFLAGS}
    #     }
    #     foreach f ${include_files} {
    #         add_files ${f} -cflags ${CFLAGS}
    #     }
    # } else {
    #     foreach f ${SRC_LIST} {
    #         add_files ${f} -cflags ${CFLAGS}
    #     }
    # }

    # add_files ${PRJ_PATH}/src/axis_lib.cpp -cflags ${CFLAGS}
    # add_files ${PRJ_PATH}/include/axis_lib.h -cflags ${CFLAGS}

    # Add Testbench Files
    if {${csim} || ${cosim}} {
        # TB Files (to avoid including multiple files with main() in them)
        add_files -tb ${PRJ_PATH}/src/tb/${TB}.cpp -cflags ${CFLAGS}
        add_files -tb ${PRJ_PATH}/include/tb/${TB}.h -cflags ${CFLAGS}
    }
}

open_solution "solution_${TOP}"
# ==============================================================================
# Set Part
# ==============================================================================
if {${reset_project}} {
    if {${use_zedboard}} {
        # ZedBoard
        set_part {xc7z020clg484-1} -tool vivado
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
            set_part {xczu9eg-ffvb1156-2-i} -tool vivado
            create_clock -period 10 -name default
        } else {
            # ZedBoard (default)
            set_part {xc7z020clg484-1} -tool vivado
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
if {${relax_ii}} {
    config_schedule -effort ${scheduler_effort} -relax_ii_for_timing=0
} else {
    config_schedule -effort ${scheduler_effort}
}

# config_sdx -target sds ;# -optimization_level 3

if {${use_zcu104_pynq}} {
    config_interface -m_axi_addr64
}

config_core DSP48 -latency 3
config_dataflow -default_channel pingpong

# ==============================================================================
# Start C-Simulation
# ==============================================================================
if {${csim}} {
    if {${build_only}} {
        csim_design -clean -O -compiler gcc -ldflags ${LDFLAGS} -argv ${ARGV} -setup
    } else {
        csim_design -clean -O -compiler gcc -ldflags ${LDFLAGS} -argv ${ARGV}
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

        set FILENAME "${REPORT_FILE_PATH}/${board_name}_${TOP}.rpt"
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
    cosim_design -trace_level port -ldflags ${LDFLAGS} -argv ${ARGV} ;#-tool auto -wave_debug

    if {${report_info}} {
        puts "================================================================"
        puts "\[INFO\] Reporting information"
        puts "================================================================"

        set REPORT_FILENAME "${REPORT_FILE_PATH}/${board_name}_${TOP}.rpt"
        set HLS_REPORT_PATH "hls_${PROJECT_NAME}/solution_${TOP}/sim/report/"
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
puts "\[INFO\] Closing project: ./hls/hls_${PROJECT_NAME}"
puts "================================================================"

exit

cd ${PRJ_PATH}