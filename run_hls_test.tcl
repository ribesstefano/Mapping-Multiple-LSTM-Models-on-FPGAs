#
# @brief      Greps a file content and writes matches to a file.
#
# @param      re     Regular expression
# @param      lines  Number of lines to report/include after the found match
# @param      fin    The fin pointer
# @param      fout   The fout pointer
#
proc grep {re lines fin write_fout fout} {
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
            if {$write_fout} {
                puts $fout $line
            }
            set cnt [expr {$cnt +1}]
        } else {
            set match false
        }
    }
}

exec mkdir -p -- C:/Users/ste/phd/hls_projects/hls_svd//hls_prj/
exec mkdir -p -- C:/Users/ste/phd/hls_projects/hls_svd//hls_prj//reports
cd C:/Users/ste/phd/hls_projects/hls_svd//hls_prj/
open_project -reset "vitis_ZedBoard_HlsKernelV"
set_top "HlsKernelV"
open_solution -flow_target vivado -reset "HlsKernelV"
set_part xc7z020clg484-1 ;# ZedBoard
create_clock -period 10 -name default
config_compile -name_max_length=12 -pipeline_style=frp -enable_auto_rewind=1
config_schedule -effort=high -relax_ii_for_timing=0
config_core DSP48 -latency 3
# Source files
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/svd.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/svd_ip.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/svd_params.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/dma/axis_lib.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/dma/svd_dma.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/dma/width_converter.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/hls_utils/adder_tree.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/hls_utils/dot_prod_dsp.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/hls_utils/hw_timer.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/kernel/gemv_kernel.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/kernel/svd_kernel.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/kernel/s_kernel.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/kernel/u_kernel.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/kernel/v_kernel.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/layers/dense/hls/dense_svd.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/layers/lstm/lstm_data_handler.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/layers/lstm/hls/lstm_hardware.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/layers/lstm/hls/lstm_svd.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/layers/lstm/hls/lstm_svd_emulator.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/layers/lstm/sw/soft_lstm.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/layers/lstm/sw/soft_lstm_svd.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/math_utils/activation_functions.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/math_utils/blas_utils.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//src/math_utils/data_handler.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files -tb C:/Users/ste/phd/hls_projects/hls_svd//src/testbenches/test_v_kernel.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
# Include files
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/svd_ip.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/svd_params.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/dma/axis_lib.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/dma/svd_dma.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/dma/width_converter.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/hls_utils/adder_tree.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/hls_utils/dot_prod_dsp.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/hls_utils/hls_debugging.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/hls_utils/hls_metaprogramming.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/hls_utils/hw_timer.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/hls_utils/priority_encoder.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/kernel/gemv_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/kernel/svd_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/kernel/s_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/kernel/u_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/kernel/v_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/layers/dense/hls/dense_svd.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/layers/lstm/lstm_data_handler.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/layers/lstm/hls/lstm_hardware.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/layers/lstm/hls/lstm_svd.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/layers/lstm/hls/lstm_svd_emulator.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/layers/lstm/sw/soft_lstm.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/layers/lstm/sw/soft_lstm_svd.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/math_utils/activation_functions.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/math_utils/blas_utils.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd//include/math_utils/data_handler.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
add_files -tb C:/Users/ste/phd/hls_projects/hls_svd//include/testbenches/test_v_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include"
csynth_design
puts "================================================================"
puts "\[INFO\] Reporting information"
puts "================================================================"        
set fin [open C:/Users/ste/phd/hls_projects/hls_svd//hls_prj/vitis_ZedBoard_HlsKernelV/solution_HlsKernelV/syn/report/HlsKernelV_csynth.rpt r]
set fout [open C:/Users/ste/phd/hls_projects/hls_svd//hls_prj//reports/vitis_ZedBoard_HlsKernelV.rpt a]
grep "== Performance Estimates" 18 $fin 0 $fout
grep "== Utilization Estimates" 20 $fin 0 $fout
close $fin
close $fout
puts "================================================================"
puts "\[INFO\] Closing project: vitis_ZedBoard_HlsKernelV"
puts "================================================================"
exit
cd C:/Users/ste/phd/hls_projects/hls_svd/
