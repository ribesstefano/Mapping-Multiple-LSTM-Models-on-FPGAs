import os
import argparse
import re

def get_tcl_utils():
    utils = '''#
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

'''
    return utils

def is_top_file(filename, top):
    found = False
    with open(filename, 'r') as f:
        for line in f:
            if top in line:
                found = True
                break
    return found

def get_dependent_files(filename):
    headers = []
    with open(filename, 'r') as f:
        for line in f:
            if '#include' in line:
                headers.append(line)
    return headers

def main():
    avail_fpgas = {
        'ZedBoard' : 'xc7z020clg484-1',
        'ZCU104' : 'xczu7ev-ffvc1156-2-e',
        'ZCU102' : 'xczu9eg-ffvb1156-2-i',
        'Arty-7' : 'xc7a100tcsg324-1'
    }

    parser = argparse.ArgumentParser(description='Generate a TCL script file for HLS projects generation.')
    parser.add_argument('top', metavar='top', type=str, help='The top HLS function name.')
    parser.add_argument('--top_file', type=str, default='', help='The filename of the top HLS function. Default: None.')

    parser.add_argument('--script_name', type=str, default='run_hls_test.tcl', help='Generated TCL filename. Default: run_hls.tcl.')
    parser.add_argument('--run_hls', action='store_true', help='Run th generated TCL file. Default: False.')

    parser.add_argument('--board', type=str, choices=avail_fpgas.keys(), default='ZedBoard', help='The target FPGA board.')
    parser.add_argument('--period', type=str, default='10', help='Clock period in ns. Default: 10.')
    # Actions
    parser.add_argument('--use_vivado_hls', action='store_true', help='Use Vivado HLS. Default is Vitis HLS.')
    parser.add_argument('--no_synthesis', action='store_true', help='Do not run synthesis.')
    parser.add_argument('--csim', action='store_true', help='Run C/C++ simulation.')
    parser.add_argument('--cosim', action='store_true', help='Run C/C++ co-simulation.')
    parser.add_argument('--export', action='store_true', help='Export IP.')
    parser.add_argument('--place_and_route', action='store_true', help='Export IP and check place-and-route.')
    parser.add_argument('--no_reset_prj', action='store_true', help='Do not reset HLS project.')
    parser.add_argument('--debug_dataflow', action='store_true', help='Resize FIFO depths for dataflow check.')
    parser.add_argument('--profile_dataflow', action='store_true', help='Resize FIFO depths for dataflow check.')

    parser.add_argument('--resize_fifos', action='store_true', help='Resize FIFO depths for dataflow check.')
    parser.add_argument('--scheduler_effort', type=str, choices=['low', 'medium', 'high'], default='high', help='The HLS scheduler effort. Default: high.')

    parser.add_argument('--tb_dir', type=str, default='testbenches', help='Testbench dir name under src files. Default: testbenches.')
    parser.add_argument('--tb_file', type=str, default='', help='Testbench file name under tb_dir. Default: None.')
    parser.add_argument('--cc', type=str, choices=['11', '14'], default='14', help='The target C++ version. Default: C++14.')
    parser.add_argument('--cflags', type=str, default='-O3 -std=c++14', help='C++ cflags. Default: -O3 -std=c++14.')
    parser.add_argument('--ldflags', type=str, default='', help='C++ linker flags. Default: None.')
    parser.add_argument('--argv', type=str, default='', help='The simulation and co-simulation arguments.')
    parser.add_argument('--src', type=str, default='src', help='The C/C++ source directory. Default: src.')
    parser.add_argument('--include', type=str, default='include', help='The C/C++ include directory. Default: include.')
    parser.add_argument('--defines', type=str, default='', help='The C/C++ defines. Default: None.')
    parser.add_argument('--defines_file', type=str, default='', help='The file containing all C/C++ defines. Default: None.')

    args = parser.parse_args()

    if args.csim and args.tb_file == '' or args.cosim and args.tb_file == '':
        parser.error('The --csim and --cosim arguments requires --tb_file.')
    # ==========================================================================
    # Setup
    # ==========================================================================
    curr_dir = os.getcwd().replace('\\', '/') + '/'
    hls_prj_dir = curr_dir + '/hls_prj/'
    hls_report_dir = hls_prj_dir + '/reports'
    hls_tool = 'vhls' if args.use_vivado_hls else 'vitis'
    prj_name = f'{hls_tool}_{args.board}_{args.top}'
    reset_string = ' -reset ' if args.no_reset_prj else ''
    # ==========================================================================
    # Adjust CFLAGS and add defines
    # ==========================================================================
    cflags = args.cflags + ' -I' + curr_dir + args.include
    if hls_tool == 'vhls':
        cflags = '-fno-builtin ' + cflags.replace('c++14', 'c++0x')
        print(f'[WARNING] Replacing C++14 with C++11 in Vivado HLS. CFLAGS: {cflags}')
    # ==========================================================================
    # Generate script file
    # ==========================================================================
    with open(args.script_name, 'w') as f:
        f.write(get_tcl_utils())
        f.write(f'exec mkdir -p -- {hls_prj_dir}\n')
        f.write(f'exec mkdir -p -- {hls_report_dir}\n')
        f.write(f'cd {hls_prj_dir}\n')
        f.write(f'open_project {reset_string} "{prj_name}"\n')
        f.write(f'set_top "{args.top}"\n')
        if not args.no_reset_prj:
            pass
            if args.csim or args.cosim:
                pass
        if hls_tool == 'vitis':
            f.write(f'open_solution -flow_target vivado {reset_string} "solution_{args.top}"\n')
        else:
            f.write(f'open_solution "solution_{args.top}"\n')
        # ======================================================================
        # Config Synthesis for current solution
        # ======================================================================
        if not args.no_reset_prj:
            f.write(f'set_part {avail_fpgas[args.board]} ;# {args.board}\n')
            f.write(f'create_clock -period {args.period} -name default\n')
            if hls_tool == 'vitis':
                f.write(f'config_compile -name_max_length=12 -pipeline_style=frp -enable_auto_rewind=1\n')
            else:
                f.write(f'config_compile -name_max_length=12\n')
            f.write(f'config_schedule -effort={args.scheduler_effort} -relax_ii_for_timing=0\n')
            f.write(f'config_core DSP48 -latency 3\n')
        # ======================================================================
        # Add only significant files to synthesis
        # ======================================================================
        # TODO: Recursively include only the files from which the top function depends on.
        headers = []
        if args.top_file:
            for fpath, subdirs, files in os.walk(args.src):
                for fname in files:
                    if args.top_file.replace('.cpp', '') == fname.replace('.cpp', '') or \
                            args.top_file.replace('.h', '') == fname.replace('.h', ''):
                        headers_decl = get_dependent_files(fpath + '/' + fname)
                        for h in headers_decl:
                            match = re.search('"(.*?)"', h)
                            if match is not None:
                                for m in match.groups():
                                    headers.append(m.replace('"', ''))
                            match = re.search('<(.*?)>', h)
                            if match is not None:
                                for m in match.groups():
                                    tmp = m.replace('<', '')
                                    tmp = tmp.replace('>', '')
                                    headers.append(tmp)
        print(headers)
        # ======================================================================
        # Add files
        # ======================================================================
        f.write(f'# Source files\n')
        for fpath, subdirs, files in os.walk(args.src):
            for fname in files:
                unix_filename = curr_dir + '/' + fpath.replace('\\', '/') + '/' + fname
                if fname.endswith('.cpp') or fname.endswith('.cc'):
                    if args.tb_dir.replace('\\', '/') not in fpath:
                        print(f'[INFO] Adding synthesis file: {unix_filename}')
                        f.write(f'add_files {unix_filename} -cflags "{cflags}"\n')
                    else:
                        if fname.startswith(args.tb_file):
                            print(f'[INFO] Adding simulation file: {unix_filename}')
                            f.write(f'add_files -tb {unix_filename} -cflags "{cflags}"\n')
        f.write(f'# Include files\n')
        for fpath, subdirs, files in os.walk(args.include):
            for fname in files:
                unix_filename = curr_dir + '/' + fpath.replace('\\', '/') + '/' + fname
                if fname.endswith('.h'):
                    if args.tb_dir.replace('\\', '/') not in fpath:
                        print(f'[INFO] Adding synthesis file: {unix_filename}')
                        f.write(f'add_files {unix_filename} -cflags "{cflags}"\n')
                    else:
                        if fname.startswith(args.tb_file):
                            print(f'[INFO] Adding simulation file: {unix_filename}')
                            f.write(f'add_files -tb {unix_filename} -cflags "{cflags}"\n')
        # ======================================================================
        # Start CSim
        # ======================================================================
        if args.csim:
            csim_cmd = 'csim_design -clean -O'
            if args.ldflags:
                csim_cmd += f' -ldflags {args.ldflags}'
            if args.argv:
                csim_cmd += f' -argv {args.argv}'
            f.write(f'{csim_cmd}\n')
        # ======================================================================
        # Run Synthesis and report
        # ======================================================================
        report_outfile = hls_report_dir + f'/{hls_tool}_{args.board}_{args.top}.rpt'
        if not args.no_synthesis:
            f.write(f'csynth_design\n')
            csynth_report = hls_prj_dir + prj_name + f'/solution_{args.top}/syn/report/{args.top}_csynth.rpt'
            report_info = f'''puts "================================================================"
puts "\[INFO\] Reporting information"
puts "================================================================"        
set fin [open {csynth_report} r]
set fout [open {report_outfile} a]
grep "== Performance Estimates" 18 $fin 0 $fout
grep "== Utilization Estimates" 20 $fin 0 $fout
close $fin
close $fout'''
            f.write(f'{report_info}\n')
        # ======================================================================
        # Run Cosim and report
        # ======================================================================
        if args.cosim:
            cosim_cmd = f'cosim_design -trace_level port -ldflags "{args.ldflags}" -argv "{args.argv}"'
            if hls_tool == 'vitis' and args.debug_dataflow:
                cosim_cmd += ' -enable_dataflow_profiling=1 -enable_fifo_sizing=1'

                
            f.write(f'{cosim_cmd}\n')

            cosim_report = hls_prj_dir + prj_name + f'/solution_{args.top}/sim/report/{args.top}_cosim.rpt'
            report_info = f'''puts "================================================================"
puts "\[INFO\] Reporting information"
puts "================================================================"        
set fin [open {cosim_report} r]
set fout [open {report_outfile} a]
grep "Simulation tool" 10 $fin 0 $fout
close $fin
close $fout'''
            f.write(f'{report_info}\n')
        # ======================================================================
        # Export IP
        # ======================================================================
        if args.export:
            f.write('export_design -format ip_catalog\n')
        if args.place_and_route:
            f.write('export_design -flow impl -rtl verilog -format ip_catalog\n')
        # ======================================================================
        # Close and run
        # ======================================================================
        closing = f'''puts "================================================================"
puts "\[INFO\] Closing project: {prj_name}"
puts "================================================================"
exit
cd {curr_dir}'''
        f.write(f'{closing}\n')
    if args.run_hls:
        if hls_tool == 'vhls':
            os.system(f'vivado_hls -f {args.script_name}')
        else:
            os.system(f'D:/Programs/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls -f {args.script_name}')

if __name__ == '__main__':
    main()