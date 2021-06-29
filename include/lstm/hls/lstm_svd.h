#ifndef LSTM_HLS_LSTM_SVD_H_
#define LSTM_HLS_LSTM_SVD_H_

#include "svd_params.h"

#ifdef SDS_DESIGN
// =============================================================================
// Ports using DMAs
// =============================================================================
#pragma SDS data copy(x1_port[0:INPUT_SIZE])
#pragma SDS data copy(x2_port[0:INPUT_SIZE])
#pragma SDS data copy(h_t1_prev_port[0:HIDDEN_SIZE])
#pragma SDS data copy(h_t2_prev_port[0:HIDDEN_SIZE])
#pragma SDS data copy(c_t1_prev_port[0:HIDDEN_SIZE])
#pragma SDS data copy(c_t2_prev_port[0:HIDDEN_SIZE])
#pragma SDS data copy(bias1_port[0:4*HIDDEN_SIZE])
#pragma SDS data copy(bias2_port[0:4*HIDDEN_SIZE])
#pragma SDS data copy(comb_v_port[0:NUM_ITERATIONS * 8])
#pragma SDS data copy(comb_u_port[0:NUM_ITERATIONS * 8])
#pragma SDS data copy(h_t1_curr_port[0:HIDDEN_SIZE])
#pragma SDS data copy(h_t2_curr_port[0:HIDDEN_SIZE])
#pragma SDS data copy(c_t1_curr_port[0:HIDDEN_SIZE])
#pragma SDS data copy(c_t2_curr_port[0:HIDDEN_SIZE])
// Data Movers
#pragma SDS data data_mover(x1_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(x2_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(h_t1_prev_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(h_t2_prev_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(c_t1_prev_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(c_t2_prev_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(bias1_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(bias2_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(comb_v_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(comb_u_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(h_t1_curr_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(h_t2_curr_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(c_t1_curr_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(c_t2_curr_port:AXIDMA_SIMPLE)
// Port mapping
// #pragma SDS data sys_port(x1_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(x2_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(h_t1_prev_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(h_t2_prev_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(c_t1_prev_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(c_t2_prev_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(bias1_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(bias2_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(comb_v_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(comb_u_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(h_t1_curr_port:ps_e_S_AXI_HPC1_FPD) // Coherent HP port
// #pragma SDS data sys_port(h_t2_curr_port:ps_e_S_AXI_HPC1_FPD) // Coherent HP port
// #pragma SDS data sys_port(c_t1_curr_port:ps_e_S_AXI_HPC1_FPD) // Coherent HP port
// #pragma SDS data sys_port(c_t2_curr_port:ps_e_S_AXI_HPC1_FPD) // Coherent HP port
// =============================================================================
// Weight ports not using DMAs
// =============================================================================
// #pragma SDS data zero_copy(u_cur_port[0:NUM_ITERATIONS*INPUT_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)])
// #pragma SDS data zero_copy(u_rec_port[0:NUM_ITERATIONS*HIDDEN_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)])
// #pragma SDS data zero_copy(v_port[0:NUM_ITERATIONS*HIDDEN_SIZE / NUM_TILES_V * (NUM_TILES_V - NUM_ZERO_TILES_V)])
// #pragma SDS data zero_copy(s1_port[0:NUM_ITERATIONS])
// #pragma SDS data zero_copy(s2_port[0:NUM_ITERATIONS])
// =============================================================================
// Weight ports using DMAs
// =============================================================================
#pragma SDS data copy(u_cur_port[0:NUM_ITERATIONS*INPUT_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)])
#pragma SDS data copy(u_rec_port[0:NUM_ITERATIONS*HIDDEN_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)])
#pragma SDS data copy(v_port[0:NUM_ITERATIONS*HIDDEN_SIZE / NUM_TILES_V * (NUM_TILES_V - NUM_ZERO_TILES_V)])
#pragma SDS data copy(s1_port[0:NUM_ITERATIONS])
#pragma SDS data copy(s2_port[0:NUM_ITERATIONS])
// Platform Port Mapping, available options:
// - ACP Coherent ports: ps_e_S_AXI_HPC[0-1]_FPD
// - HP ports: ps_e_S_AXI_HP[0-3]_FPD
// #pragma SDS data sys_port(u_cur_port:ps_e_S_AXI_HP0_FPD) // HP2
// #pragma SDS data sys_port(u_rec_port:ps_e_S_AXI_HP1_FPD) // HP3
// #pragma SDS data sys_port(v_port:ps_e_S_AXI_HP2_FPD) // HP3
// #pragma SDS data sys_port(s1_port:ps_e_S_AXI_HP3_FPD) // HP3
// #pragma SDS data sys_port(s2_port:ps_e_S_AXI_HP3_FPD) // HP3
// =============================================================================
// Other Configurations
// =============================================================================
// Compiler hint on allocation
#pragma SDS data mem_attribute(x1_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(x2_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(h_t1_prev_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(h_t2_prev_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_t1_prev_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_t2_prev_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(u_cur_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(u_rec_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(v_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(s1_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(s2_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(bias1_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(bias2_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(comb_v_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(comb_u_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(h_t1_curr_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(h_t2_curr_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_t1_curr_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_t2_curr_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
// NOTE: All ports are accessed sequentially.
#pragma SDS data access_pattern(x1_port:SEQUENTIAL)
#pragma SDS data access_pattern(x2_port:SEQUENTIAL)
#pragma SDS data access_pattern(h_t1_prev_port:SEQUENTIAL)
#pragma SDS data access_pattern(h_t2_prev_port:SEQUENTIAL)
#pragma SDS data access_pattern(c_t1_prev_port:SEQUENTIAL)
#pragma SDS data access_pattern(c_t2_prev_port:SEQUENTIAL)
#pragma SDS data access_pattern(u_cur_port:SEQUENTIAL)
#pragma SDS data access_pattern(u_rec_port:SEQUENTIAL)
#pragma SDS data access_pattern(v_port:SEQUENTIAL)
#pragma SDS data access_pattern(s1_port:SEQUENTIAL)
#pragma SDS data access_pattern(s2_port:SEQUENTIAL)
#pragma SDS data access_pattern(bias1_port:SEQUENTIAL)
#pragma SDS data access_pattern(bias2_port:SEQUENTIAL)
#pragma SDS data access_pattern(comb_v_port:SEQUENTIAL)
#pragma SDS data access_pattern(comb_u_port:SEQUENTIAL)
#pragma SDS data access_pattern(h_t1_curr_port:SEQUENTIAL)
#pragma SDS data access_pattern(h_t2_curr_port:SEQUENTIAL)
#pragma SDS data access_pattern(c_t1_curr_port:SEQUENTIAL)
#pragma SDS data access_pattern(c_t2_curr_port:SEQUENTIAL)
#endif // end SDS_DESIGN
void SvdModel2LstmSDSoCV2(
    const svd::ActivationD x1_port[INPUT_SIZE],
    const svd::ActivationD x2_port[INPUT_SIZE],
    const svd::ActivationD h_t1_prev_port[HIDDEN_SIZE],
    const svd::ActivationD h_t2_prev_port[HIDDEN_SIZE],
    const svd::ActivationD c_t1_prev_port[HIDDEN_SIZE],
    const svd::ActivationD c_t2_prev_port[HIDDEN_SIZE],
    const ap_uint<FIX_WIDTH * 4> *u_cur_port, // [NUM_ITERATIONS*4*INPUT_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)],
    const ap_uint<FIX_WIDTH * 4> *u_rec_port, // [NUM_ITERATIONS*4*HIDDEN_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)],
    const ap_uint<FIX_WIDTH * 8> *v_port, // [NUM_ITERATIONS*4*2*HIDDEN_SIZE / NUM_TILES_V * (NUM_TILES_V - NUM_ZERO_TILES_V)],
    const ap_uint<FIX_WIDTH * 8> *s1_port, // [NUM_ITERATIONS*8],
    const ap_uint<FIX_WIDTH * 8> *s2_port, // [NUM_ITERATIONS*8],
    const svd::WeightD bias1_port[4 * HIDDEN_SIZE],
    const svd::WeightD bias2_port[4 * HIDDEN_SIZE],
    const ap_uint<NUM_TILES_V> comb_v_port[NUM_ITERATIONS * 8],
    const ap_uint<NUM_TILES_U> comb_u_port[NUM_ITERATIONS * 8],
    svd::ActivationD h_t1_curr_port[HIDDEN_SIZE],
    svd::ActivationD h_t2_curr_port[HIDDEN_SIZE],
    svd::ActivationD c_t1_curr_port[HIDDEN_SIZE],
    svd::ActivationD c_t2_curr_port[HIDDEN_SIZE]);

#endif // end LSTM_HLS_LSTM_SVD_H_