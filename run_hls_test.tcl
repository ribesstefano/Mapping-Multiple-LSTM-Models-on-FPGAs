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
open_project  -reset  "vitis_ZedBoard_HlsKernelU"
set_top "HlsKernelU"
open_solution -flow_target vivado  -reset  "solution_HlsKernelU"
set_part xc7z020clg484-1 ;# ZedBoard
create_clock -period 10 -name default
config_compile -name_max_length=12 -pipeline_style=frp -enable_auto_rewind=1
config_schedule -effort=high -relax_ii_for_timing=0
config_core DSP48 -latency 3
# Synthesis files
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/kernel/u_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/kernel/gemv_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/hls_utils/adder_tree.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/dma/svd_dma.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/dma/axis_lib.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/kernel/u_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/src/kernel/u_kernel.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/kernel/gemv_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/src/kernel/gemv_kernel.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/hls_utils/adder_tree.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/src/hls_utils/adder_tree.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/dma/svd_dma.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/src/dma/svd_dma.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/dma/axis_lib.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/src/dma/axis_lib.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/svd_params.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/src/svd_params.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/hls_utils/hls_metaprogramming.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/hls_utils/hls_debugging.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/hls_utils/priority_encoder.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/dma/width_converter.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/src/dma/width_converter.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/hls_utils/hw_timer.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/src/hls_utils/hw_timer.cpp -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/kernel/u_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/kernel/gemv_kernel.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/hls_utils/adder_tree.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/dma/svd_dma.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/dma/axis_lib.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/svd_params.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/hls_utils/hls_metaprogramming.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/hls_utils/hls_debugging.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/hls_utils/priority_encoder.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/dma/width_converter.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
add_files C:/Users/ste/phd/hls_projects/hls_svd/include/hls_utils/hw_timer.h -cflags "-O3 -std=c++14 -IC:/Users/ste/phd/hls_projects/hls_svd/include -I/usr/local/include"
csynth_design
puts "================================================================"
puts "\[INFO\] Reporting information"
puts "================================================================"        
set fin [open C:/Users/ste/phd/hls_projects/hls_svd//hls_prj/vitis_ZedBoard_HlsKernelU/solution_HlsKernelU/syn/report/HlsKernelU_csynth.rpt r]
set fout [open C:/Users/ste/phd/hls_projects/hls_svd//hls_prj//reports/vitis_ZedBoard_HlsKernelU.rpt a]
grep "== Performance Estimates" 18 $fin 0 $fout
grep "== Utilization Estimates" 20 $fin 0 $fout
close $fin
close $fout
puts "================================================================"
puts "\[INFO\] Closing project: vitis_ZedBoard_HlsKernelU"
puts "================================================================"
exit
cd C:/Users/ste/phd/hls_projects/hls_svd/
