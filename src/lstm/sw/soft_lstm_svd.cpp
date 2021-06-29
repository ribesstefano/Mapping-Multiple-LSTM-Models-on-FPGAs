#include "lstm/sw/soft_lstm_svd.h"

#ifdef __cplusplus
extern "C"
#endif
void SvdModelSoftwareBatched(const int verbose,
                             const bool use_blas,
                             const float *x,
                             const int num_samples,
                             const int num_timesteps,
                             const int n_steps,
                             const int input_size,
                             const int hidden_size,
                             const float *cur_i_u,
                             const float *cur_i_s,
                             const float *cur_i_v,
                             const float *cur_f_u,
                             const float *cur_f_s,
                             const float *cur_f_v,
                             const float *cur_c_u,
                             const float *cur_c_s,
                             const float *cur_c_v,
                             const float *cur_o_u,
                             const float *cur_o_s,
                             const float *cur_o_v,
                             const float *rec_i_u,
                             const float *rec_i_s,
                             const float *rec_i_v,
                             const float *rec_f_u,
                             const float *rec_f_s,
                             const float *rec_f_v,
                             const float *rec_c_u,
                             const float *rec_c_s,
                             const float *rec_c_v,
                             const float *rec_o_u,
                             const float *rec_o_s,
                             const float *rec_o_v,
                             const float *bias_i,
                             const float *bias_f,
                             const float *bias_c,
                             const float *bias_o,
                             float *out) {
  // ===========================================================================
  // This C++ function is implementing the following python snippet:
  // (Note the extensive use of matrix multiplications)
  // 
  //   m = 6 # input size
  //   n = 8 # output size, i.e. LSTM hidden size (in Keras)
  //   b = 4 # batch size (or timesteps)
  //   k = 2 # num iterations, i.e. n_steps
  //   x = np.random.randn(m, b)
  //   u = np.random.randn(m, k)
  //   v = np.random.randn(n, k)
  //   s = np.random.randn(k)
  //   s2 = np.random.randn(k)
  //   
  //   s_mat = np.repeat(s, m).reshape(k, m).T
  //   us = u * s_mat
  //   ux = x.T @ us
  //   y_batched = ux @ v.T
  // ===========================================================================
  // ===========================================================================
  // Define: x.T @ US Matrixes
  // ===========================================================================
  // Current x
  float *cur_i_ux = new float[num_timesteps * n_steps];
  float *cur_f_ux = new float[num_timesteps * n_steps];
  float *cur_c_ux = new float[num_timesteps * n_steps];
  float *cur_o_ux = new float[num_timesteps * n_steps];
  // Recurrent x
  float *rec_i_uh = new float[n_steps];
  float *rec_f_uh = new float[n_steps];
  float *rec_c_uh = new float[n_steps];
  float *rec_o_uh = new float[n_steps];

  // Current y
  float *cur_i_y = new float[num_timesteps * hidden_size];
  float *cur_f_y = new float[num_timesteps * hidden_size];
  float *cur_c_y = new float[num_timesteps * hidden_size];
  float *cur_o_y = new float[num_timesteps * hidden_size];
  // Recurrent y
  float *rec_i_y = new float[hidden_size];
  float *rec_f_y = new float[hidden_size];
  float *rec_c_y = new float[hidden_size];
  float *rec_o_y = new float[hidden_size];

  // ===========================================================================
  // NOTE: Broadcasting bias vector into a matrix for exploiting C initialization
  // in gemm function (C = A * B + C) produces no significant gain in terms of
  // execution time and makes the code harder to read...
  // ===========================================================================
  // Output y + bias
  float *i_cur_bias = new float[num_timesteps * hidden_size];
  float *f_cur_bias = new float[num_timesteps * hidden_size];
  float *c_cur_bias = new float[num_timesteps * hidden_size];
  float *o_cur_bias = new float[num_timesteps * hidden_size];

  float *i_sum = new float[hidden_size];
  float *f_sum = new float[hidden_size];
  float *c_sum = new float[hidden_size];
  float *o_sum = new float[hidden_size];

  float *i_gate = new float[hidden_size];
  float *f_gate = new float[hidden_size];
  float *o_gate = new float[hidden_size];
  float *c_sum_tanh = new float[hidden_size];
  float *c_tanh = new float[hidden_size];
  float *c_lhs = new float[hidden_size];
  float *c_rhs = new float[hidden_size];
  float *c = new float[hidden_size];

  // Recurrent v Transposed
  float *rec_i_v_T = new float[hidden_size * n_steps];
  float *rec_f_v_T = new float[hidden_size * n_steps];
  float *rec_c_v_T = new float[hidden_size * n_steps];
  float *rec_o_v_T = new float[hidden_size * n_steps];
  // ===========================================================================
  // Compute:
  // - S * U Matrix
  // ===========================================================================
  // Current us1
  float *cur_i_u_T = new float[n_steps * input_size];
  float *cur_f_u_T = new float[n_steps * input_size];
  float *cur_c_u_T = new float[n_steps * input_size];
  float *cur_o_u_T = new float[n_steps * input_size];
  // Recurrent us
  float *rec_i_u_T = new float[n_steps * hidden_size];
  float *rec_f_u_T = new float[n_steps * hidden_size];
  float *rec_c_u_T = new float[n_steps * hidden_size];
  float *rec_o_u_T = new float[n_steps * hidden_size];
  // Current us1
  float *cur_i_us = new float[n_steps * input_size];
  float *cur_f_us = new float[n_steps * input_size];
  float *cur_c_us = new float[n_steps * input_size];
  float *cur_o_us = new float[n_steps * input_size];
  // Recurrent us
  float *rec_i_us = new float[n_steps * hidden_size];
  float *rec_f_us = new float[n_steps * hidden_size];
  float *rec_c_us = new float[n_steps * hidden_size];
  float *rec_o_us = new float[n_steps * hidden_size];

  // ===========================================================================
  // NOTE: We need to 'transpose' u in order to generate the us matrix. This is
  // because the multiplication svd_mul() operates row-wise.
  // ===========================================================================
  // BEFORE TRANSPOSE: s.shape = (n_steps)
  // BEFORE TRANSPOSE: u.shape = (n_steps, input_size)
  // BEFORE TRANSPOSE: us.shape = (n_steps, input_size)
  svd_transpose(n_steps, input_size, cur_i_u, cur_i_u_T);
  svd_transpose(n_steps, input_size, cur_f_u, cur_f_u_T);
  svd_transpose(n_steps, input_size, cur_c_u, cur_c_u_T);
  svd_transpose(n_steps, input_size, cur_o_u, cur_o_u_T);
  svd_transpose(n_steps, hidden_size, rec_i_u, rec_i_u_T);
  svd_transpose(n_steps, hidden_size, rec_f_u, rec_f_u_T);
  svd_transpose(n_steps, hidden_size, rec_c_u, rec_c_u_T);
  svd_transpose(n_steps, hidden_size, rec_o_u, rec_o_u_T);
  for (int i = 0; i < input_size; ++i) {
    svd_mul(n_steps, &cur_i_u_T[i*n_steps], cur_i_s, &cur_i_us[i*n_steps]);
    svd_mul(n_steps, &cur_f_u_T[i*n_steps], cur_f_s, &cur_f_us[i*n_steps]);
    svd_mul(n_steps, &cur_c_u_T[i*n_steps], cur_c_s, &cur_c_us[i*n_steps]);
    svd_mul(n_steps, &cur_o_u_T[i*n_steps], cur_o_s, &cur_o_us[i*n_steps]);
  }
  for (int i = 0; i < hidden_size; ++i) {
    svd_mul(n_steps, &rec_i_u_T[i*n_steps], rec_i_s, &rec_i_us[i*n_steps]);
    svd_mul(n_steps, &rec_f_u_T[i*n_steps], rec_f_s, &rec_f_us[i*n_steps]);
    svd_mul(n_steps, &rec_c_u_T[i*n_steps], rec_c_s, &rec_c_us[i*n_steps]);
    svd_mul(n_steps, &rec_o_u_T[i*n_steps], rec_o_s, &rec_o_us[i*n_steps]);
  }
  // ===========================================================================
  // Transpose recurrent v and recurrent u vectors.
  // ===========================================================================
  // From (hidden_size, n_steps) to (n_steps, hidden_size)
  svd_transpose(hidden_size, n_steps, rec_i_us, rec_i_u_T);
  svd_transpose(hidden_size, n_steps, rec_f_us, rec_f_u_T);
  svd_transpose(hidden_size, n_steps, rec_c_us, rec_c_u_T);
  svd_transpose(hidden_size, n_steps, rec_o_us, rec_o_u_T);
  // From (n_steps, hidden_size) to (hidden_size, n_steps)
  svd_transpose(n_steps, hidden_size, rec_i_v, rec_i_v_T);
  svd_transpose(n_steps, hidden_size, rec_f_v, rec_f_v_T);
  svd_transpose(n_steps, hidden_size, rec_c_v, rec_c_v_T);
  svd_transpose(n_steps, hidden_size, rec_o_v, rec_o_v_T);

#ifdef USE_BLAS
  const bool kAvailableBLAS = true;
#else
  const bool kAvailableBLAS = false;
#endif

#ifdef SDS_DESIGN
  perf_counter sw_ctr;
  perf_counter gemm_ctr;
  perf_counter gemv_ctr;
  sw_ctr.start();
#elif !defined(__SYNTHESIS__)
  auto begin = std::chrono::high_resolution_clock::now();
#endif

  if (use_blas && kAvailableBLAS) {
#ifdef USE_BLAS
    for (int i = 0; i < num_samples; ++i) {
#ifdef SDS_DESIGN
      gemm_ctr.start();
#endif
      // Current x @ us: (num_timesteps, input_size) @ (input_size, n_steps)
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, n_steps, input_size, (float)1., &x[i * input_size * num_timesteps], cur_i_us, (float)0., cur_i_ux);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, n_steps, input_size, (float)1., &x[i * input_size * num_timesteps], cur_f_us, (float)0., cur_f_ux);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, n_steps, input_size, (float)1., &x[i * input_size * num_timesteps], cur_c_us, (float)0., cur_c_ux);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, n_steps, input_size, (float)1., &x[i * input_size * num_timesteps], cur_o_us, (float)0., cur_o_ux);

      // Current ux @ v: (num_timesteps, n_steps) @ (n_steps, hidden_size)
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, hidden_size, n_steps, (float)1., cur_i_ux, cur_i_v, (float)0., cur_i_y);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, hidden_size, n_steps, (float)1., cur_f_ux, cur_f_v, (float)0., cur_f_y);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, hidden_size, n_steps, (float)1., cur_c_ux, cur_c_v, (float)0., cur_c_y);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, hidden_size, n_steps, (float)1., cur_o_ux, cur_o_v, (float)0., cur_o_y);

#ifdef SDS_DESIGN
      gemm_ctr.stop();
#endif

      svd_set(hidden_size, (float)0., c);
      svd_set(hidden_size, (float)0., &out[i * hidden_size]);

      for (int j = 0; j < num_timesteps; ++j) {
#ifdef SDS_DESIGN
        gemv_ctr.start();
#endif
        // Recurrent us.T @ h: (hidden_size, n_steps).T @ (hidden_size, 1)
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_i_u_T, &out[i * hidden_size], (float)0., rec_i_uh);
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_f_u_T, &out[i * hidden_size], (float)0., rec_f_uh);
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_c_u_T, &out[i * hidden_size], (float)0., rec_c_uh);
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_o_u_T, &out[i * hidden_size], (float)0., rec_o_uh);

        // Recurrent v.T @ uh: (n_steps, hidden_size).T @ (n_steps, 1)
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_i_v_T, rec_i_uh, (float)0., rec_i_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_f_v_T, rec_f_uh, (float)0., rec_f_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_c_v_T, rec_c_uh, (float)0., rec_c_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_o_v_T, rec_o_uh, (float)0., rec_o_y);

#ifdef SDS_DESIGN
        gemv_ctr.stop();
#endif
        // =======================================================================
        // Non Linearities
        // =======================================================================
        svd_add(hidden_size, &cur_i_y[j * hidden_size], bias_i, i_cur_bias);
        svd_add(hidden_size, &cur_f_y[j * hidden_size], bias_f, f_cur_bias);
        svd_add(hidden_size, &cur_c_y[j * hidden_size], bias_c, c_cur_bias);
        svd_add(hidden_size, &cur_o_y[j * hidden_size], bias_o, o_cur_bias);

        svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
        svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
        svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
        svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

        svd_hard_sigmoid(hidden_size, i_sum, i_gate);
        svd_hard_sigmoid(hidden_size, f_sum, f_gate);
        svd_hard_sigmoid(hidden_size, o_sum, o_gate);
        svd_tanh(hidden_size, c_sum, c_sum_tanh);
        svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
        svd_mul(hidden_size, c, f_gate, c_rhs);

        svd_add(hidden_size, c_lhs, c_rhs, c);
        svd_tanh(hidden_size, c, c_tanh);
        svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
      }
    } 
#endif // end USE_BLAS
  } else {
    // =========================================================================
    // @todo (11/02/2019): openmp NOT WORKING: threads still share resources,
    // creating data races and producing incorrect results.
    // =========================================================================
    // #pragma omp parallel for schedule(static, 8)
    // =========================================================================
// #pragma omp parallel for num_threads(8) schedule(dynamic)
    for (int i = 0; i < num_samples; ++i) {
      hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_i_us, cur_i_ux);
      hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_f_us, cur_f_ux);
      hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_c_us, cur_c_ux);
      hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_o_us, cur_o_ux);

      hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_i_ux, cur_i_v, cur_i_y);
      hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_f_ux, cur_f_v, cur_f_y);
      hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_c_ux, cur_c_v, cur_c_y);
      hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_o_ux, cur_o_v, cur_o_y);

      svd_set(hidden_size, (float)0., c);
      svd_set(hidden_size, (float)0., &out[i * hidden_size]);

      for (int j = 0; j < num_timesteps; ++j) {
        hls_gemv<float, float>(n_steps, hidden_size, rec_i_u_T, &out[i*hidden_size], rec_i_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_f_u_T, &out[i*hidden_size], rec_f_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_c_u_T, &out[i*hidden_size], rec_c_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_o_u_T, &out[i*hidden_size], rec_o_uh);

        hls_gemv<float, float>(hidden_size, n_steps, rec_i_v_T, rec_i_uh, rec_i_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_f_v_T, rec_f_uh, rec_f_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_c_v_T, rec_c_uh, rec_c_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_o_v_T, rec_o_uh, rec_o_y);
        // =======================================================================
        // Non Linearities
        // =======================================================================
        svd_add(hidden_size, &cur_i_y[j * hidden_size], bias_i, i_cur_bias);
        svd_add(hidden_size, &cur_f_y[j * hidden_size], bias_f, f_cur_bias);
        svd_add(hidden_size, &cur_c_y[j * hidden_size], bias_c, c_cur_bias);
        svd_add(hidden_size, &cur_o_y[j * hidden_size], bias_o, o_cur_bias);

        svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
        svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
        svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
        svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

        svd_hard_sigmoid(hidden_size, i_sum, i_gate);
        svd_hard_sigmoid(hidden_size, f_sum, f_gate);
        svd_hard_sigmoid(hidden_size, o_sum, o_gate);
        svd_tanh(hidden_size, c_sum, c_sum_tanh);
        svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
        svd_mul(hidden_size, c, f_gate, c_rhs);

        svd_add(hidden_size, c_lhs, c_rhs, c);
        svd_tanh(hidden_size, c, c_tanh);
        svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
      }
    }
  }
  // ===========================================================================
  // NOTE: We are NOT taking into account the time it takes to both setup the u @ s
  // matrices and perform their transpositions because these operations can be
  // done "offline", i.e. can be stored in that form already, performance-wise.
  // ===========================================================================
#ifdef SDS_DESIGN
  sw_ctr.stop();
#elif !defined(__SYNTHESIS__)
  auto end = std::chrono::high_resolution_clock::now();
#endif

  if (verbose == 1) {
#ifdef SDS_DESIGN
    auto sw_cycles = sw_ctr.avg_cpu_cycles();
    auto sw_freq = sds_clock_frequency();
    std::cout << "Frequency: " << sw_freq << " ticks/second\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
    std::cout << "Batched SVD: Total CPU cycles: " << std::setprecision(12)
              << sw_cycles << std::setprecision(6) << "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
    std::cout << "Batched SVD: Average CPU cycles per sample: "
              << sw_cycles / num_samples << "\n";
    std::cout << "Batched SVD: Average CPU cycles per timesteps: "
              << sw_cycles / num_samples / num_timesteps << "\n";


    auto gemm_cycles = gemm_ctr.avg_cpu_cycles();
    if (use_blas && kAvailableBLAS) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
    std::cout << "Batched SVD-GEMM: Total CPU cycles: " << std::setprecision(12)
              << gemm_cycles << std::setprecision(6) << "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
    std::cout << "Batched SVD-GEMM: Average CPU cycles per sample: "
              << gemm_cycles / num_samples << "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
    std::cout << "Batched SVD-GEMM: Average CPU cycles per timesteps: "
              << gemm_cycles / num_samples / num_timesteps << "\n";

    auto gemv_cycles = gemv_ctr.avg_cpu_cycles();
    if (use_blas && kAvailableBLAS) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
    std::cout << "Batched SVD-GEMV: Total CPU cycles: " << std::setprecision(12)
              << gemv_cycles << std::setprecision(6) << "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
    std::cout << "Batched SVD-GEMV: Average CPU cycles per sample: "
              << gemv_cycles / num_samples << "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
    std::cout << "Batched SVD-GEMV: Average CPU cycles per timesteps: "
              << gemv_cycles / num_samples / num_timesteps << "\n";

#elif !defined(__SYNTHESIS__)
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - 
      begin).count();
    auto duration_us = duration_ns / 1000.0;
    auto duration_ms = duration_us / 1000.0;
    auto duration_s = duration_ms / 1000.0;
    if (use_blas && kAvailableBLAS) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
    std::cout << "Batched SVD: Total time: " << duration_ms
              << " ms (" << duration_s << " s)."<< "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
    std::cout << "Batched SVD: Average time per sample: " << duration_ms / num_samples
              << " ms (" << duration_s / num_samples << " s)."<< "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
    std::cout << "Batched SVD: Average time per timesteps: " << duration_ms / num_samples / num_timesteps
              << " ms (" << duration_s / num_samples / num_timesteps << " s)."<< "\n";
#endif
  }
  delete[] cur_i_ux;
  delete[] cur_f_ux;
  delete[] cur_c_ux;
  delete[] cur_o_ux;
  delete[] rec_i_uh;
  delete[] rec_f_uh;
  delete[] rec_c_uh;
  delete[] rec_o_uh;
  delete[] cur_i_y;
  delete[] cur_f_y;
  delete[] cur_c_y;
  delete[] cur_o_y;
  delete[] rec_i_y;
  delete[] rec_f_y;
  delete[] rec_c_y;
  delete[] rec_o_y;
  delete[] i_cur_bias;
  delete[] f_cur_bias;
  delete[] c_cur_bias;
  delete[] o_cur_bias;
  delete[] i_sum;
  delete[] f_sum;
  delete[] c_sum;
  delete[] o_sum;
  delete[] i_gate;
  delete[] f_gate;
  delete[] o_gate;
  delete[] c_sum_tanh;
  delete[] c_tanh;
  delete[] c_lhs;
  delete[] c_rhs;
  delete[] c;
  delete[] rec_i_v_T;
  delete[] rec_f_v_T;
  delete[] rec_c_v_T;
  delete[] rec_o_v_T;
  delete[] cur_i_u_T;
  delete[] cur_f_u_T;
  delete[] cur_c_u_T;
  delete[] cur_o_u_T;
  delete[] rec_i_u_T;
  delete[] rec_f_u_T;
  delete[] rec_c_u_T;
  delete[] rec_o_u_T;
  delete[] cur_i_us;
  delete[] cur_f_us;
  delete[] cur_c_us;
  delete[] cur_o_us;
  delete[] rec_i_us;
  delete[] rec_f_us;
  delete[] rec_c_us;
  delete[] rec_o_us;
}

#ifdef __cplusplus
extern "C"
#endif
void SvdModelSoftwareUnbatched(const int verbose,
                               const bool use_blas,
                               const float *x,
                               const int num_samples,
                               const int num_timesteps,
                               const int n_steps,
                               const int input_size,
                               const int hidden_size,
                               const float *cur_i_u,
                               const float *cur_i_s,
                               const float *cur_i_v,
                               const float *cur_f_u,
                               const float *cur_f_s,
                               const float *cur_f_v,
                               const float *cur_c_u,
                               const float *cur_c_s,
                               const float *cur_c_v,
                               const float *cur_o_u,
                               const float *cur_o_s,
                               const float *cur_o_v,
                               const float *rec_i_u,
                               const float *rec_i_s,
                               const float *rec_i_v,
                               const float *rec_f_u,
                               const float *rec_f_s,
                               const float *rec_f_v,
                               const float *rec_c_u,
                               const float *rec_c_s,
                               const float *rec_c_v,
                               const float *rec_o_u,
                               const float *rec_o_s,
                               const float *rec_o_v,
                               const float *bias_i,
                               const float *bias_f,
                               const float *bias_c,
                               const float *bias_o,
                               float *out) {
  // ===========================================================================
  // This C++ function is implementing the following python snippet:
  // (Note the extensive use of matrix multiplications)
  // 
  //   m = 6 # input size
  //   n = 8 # output size, i.e. LSTM hidden size (in Keras)
  //   b = 4 # batch size (or timesteps)
  //   k = 2 # num iterations, i.e. n_steps
  //   x = np.random.randn(m, b)
  //   u = np.random.randn(m, k)
  //   v = np.random.randn(n, k)
  //   s = np.random.randn(k)
  //   s2 = np.random.randn(k)
  //   
  //   s_mat = np.repeat(s, m).reshape(k, m).T
  //   us = u * s_mat
  //   
  //   y_not_batched = np.zeros((b, n))
  //   for i in range(b):
  //       ux = x.T[i] @ us
  //       y_not_batched[i] = ux @ v.T
  // ===========================================================================
  // ===========================================================================
  // Define: x.T @ US Matrixes
  // ===========================================================================
  // Current x
  float *cur_i_ux = new float[n_steps];
  float *cur_f_ux = new float[n_steps];
  float *cur_c_ux = new float[n_steps];
  float *cur_o_ux = new float[n_steps];
  // Recurrent x
  float *rec_i_uh = new float[n_steps];
  float *rec_f_uh = new float[n_steps];
  float *rec_c_uh = new float[n_steps];
  float *rec_o_uh = new float[n_steps];

  // Current y
  float *cur_i_y = new float[hidden_size];
  float *cur_f_y = new float[hidden_size];
  float *cur_c_y = new float[hidden_size];
  float *cur_o_y = new float[hidden_size];
  // Recurrent y
  float *rec_i_y = new float[hidden_size];
  float *rec_f_y = new float[hidden_size];
  float *rec_c_y = new float[hidden_size];
  float *rec_o_y = new float[hidden_size];

  // Output y
  float *i_cur_bias = new float[hidden_size];
  float *f_cur_bias = new float[hidden_size];
  float *c_cur_bias = new float[hidden_size];
  float *o_cur_bias = new float[hidden_size];

  float *i_sum = new float[hidden_size];
  float *f_sum = new float[hidden_size];
  float *c_sum = new float[hidden_size];
  float *o_sum = new float[hidden_size];

  float *i_gate = new float[hidden_size];
  float *f_gate = new float[hidden_size];
  float *o_gate = new float[hidden_size];
  float *c_sum_tanh = new float[hidden_size];
  float *c_tanh = new float[hidden_size];
  float *c_lhs = new float[hidden_size];
  float *c_rhs = new float[hidden_size];
  float *c = new float[hidden_size];

  // Current v Transposed
  float *cur_i_v_T = new float[hidden_size * n_steps];
  float *cur_f_v_T = new float[hidden_size * n_steps];
  float *cur_c_v_T = new float[hidden_size * n_steps];
  float *cur_o_v_T = new float[hidden_size * n_steps];
  // Recurrent v Transposed
  float *rec_i_v_T = new float[hidden_size * n_steps];
  float *rec_f_v_T = new float[hidden_size * n_steps];
  float *rec_c_v_T = new float[hidden_size * n_steps];
  float *rec_o_v_T = new float[hidden_size * n_steps];
  // ===========================================================================
  // Compute:
  // - S * U Matrix
  // ===========================================================================
  // Current us1
  float *cur_i_u_T = new float[n_steps * input_size];
  float *cur_f_u_T = new float[n_steps * input_size];
  float *cur_c_u_T = new float[n_steps * input_size];
  float *cur_o_u_T = new float[n_steps * input_size];
  // Recurrent us
  float *rec_i_u_T = new float[n_steps * hidden_size];
  float *rec_f_u_T = new float[n_steps * hidden_size];
  float *rec_c_u_T = new float[n_steps * hidden_size];
  float *rec_o_u_T = new float[n_steps * hidden_size];
  // Current us1
  float *cur_i_us = new float[n_steps * input_size];
  float *cur_f_us = new float[n_steps * input_size];
  float *cur_c_us = new float[n_steps * input_size];
  float *cur_o_us = new float[n_steps * input_size];
  // Recurrent us
  float *rec_i_us = new float[n_steps * hidden_size];
  float *rec_f_us = new float[n_steps * hidden_size];
  float *rec_c_us = new float[n_steps * hidden_size];
  float *rec_o_us = new float[n_steps * hidden_size];

  // ===========================================================================
  // NOTE: We need to 'transpose' u in order to generate the us matrix. This is
  // because the multiplication svd_mul() operates row-wise.
  // ===========================================================================
  // BEFORE TRANSPOSE: s.shape = (n_steps)
  // BEFORE TRANSPOSE: u.shape = (n_steps, input_size)
  // BEFORE TRANSPOSE: us.shape = (n_steps, input_size)
  svd_transpose(n_steps, input_size, cur_i_u, cur_i_u_T);
  svd_transpose(n_steps, input_size, cur_f_u, cur_f_u_T);
  svd_transpose(n_steps, input_size, cur_c_u, cur_c_u_T);
  svd_transpose(n_steps, input_size, cur_o_u, cur_o_u_T);
  svd_transpose(n_steps, hidden_size, rec_i_u, rec_i_u_T);
  svd_transpose(n_steps, hidden_size, rec_f_u, rec_f_u_T);
  svd_transpose(n_steps, hidden_size, rec_c_u, rec_c_u_T);
  svd_transpose(n_steps, hidden_size, rec_o_u, rec_o_u_T);
  for (int i = 0; i < input_size; ++i) {
    svd_mul(n_steps, &cur_i_u_T[i * n_steps], cur_i_s, &cur_i_us[i * n_steps]);
    svd_mul(n_steps, &cur_f_u_T[i * n_steps], cur_f_s, &cur_f_us[i * n_steps]);
    svd_mul(n_steps, &cur_c_u_T[i * n_steps], cur_c_s, &cur_c_us[i * n_steps]);
    svd_mul(n_steps, &cur_o_u_T[i * n_steps], cur_o_s, &cur_o_us[i * n_steps]);
  }
  for (int i = 0; i < hidden_size; ++i) {
    svd_mul(n_steps, &rec_i_u_T[i * n_steps], rec_i_s, &rec_i_us[i * n_steps]);
    svd_mul(n_steps, &rec_f_u_T[i * n_steps], rec_f_s, &rec_f_us[i * n_steps]);
    svd_mul(n_steps, &rec_c_u_T[i * n_steps], rec_c_s, &rec_c_us[i * n_steps]);
    svd_mul(n_steps, &rec_o_u_T[i * n_steps], rec_o_s, &rec_o_us[i * n_steps]);
  }
  // ===========================================================================
  // Transpose back current v and current u vectors.
  // ===========================================================================
  // From (input_size, n_steps) to (n_steps, input_size)
  svd_transpose(input_size, n_steps, cur_i_us, cur_i_u_T);
  svd_transpose(input_size, n_steps, cur_f_us, cur_f_u_T);
  svd_transpose(input_size, n_steps, cur_c_us, cur_c_u_T);
  svd_transpose(input_size, n_steps, cur_o_us, cur_o_u_T);
  // From (n_steps, hidden_size) to (hidden_size, n_steps)
  svd_transpose(n_steps, hidden_size, cur_i_v, cur_i_v_T);
  svd_transpose(n_steps, hidden_size, cur_f_v, cur_f_v_T);
  svd_transpose(n_steps, hidden_size, cur_c_v, cur_c_v_T);
  svd_transpose(n_steps, hidden_size, cur_o_v, cur_o_v_T); 
  // ===========================================================================
  // Transpose back recurrent v and recurrent u vectors.
  // ===========================================================================
  // From (hidden_size, n_steps) to (n_steps, hidden_size)
  svd_transpose(hidden_size, n_steps, rec_i_us, rec_i_u_T);
  svd_transpose(hidden_size, n_steps, rec_f_us, rec_f_u_T);
  svd_transpose(hidden_size, n_steps, rec_c_us, rec_c_u_T);
  svd_transpose(hidden_size, n_steps, rec_o_us, rec_o_u_T);
  // From (n_steps, hidden_size) to (hidden_size, n_steps)
  svd_transpose(n_steps, hidden_size, rec_i_v, rec_i_v_T);
  svd_transpose(n_steps, hidden_size, rec_f_v, rec_f_v_T);
  svd_transpose(n_steps, hidden_size, rec_c_v, rec_c_v_T);
  svd_transpose(n_steps, hidden_size, rec_o_v, rec_o_v_T);

  const int kSampleSize = num_timesteps * input_size;

#ifdef SDS_DESIGN
  perf_counter gemm_ctr;
  perf_counter gemv_ctr;
  perf_counter sw_ctr;
#else
#ifndef __SYNTHESIS__
  std::chrono::duration<double> total_time(0);
#endif
#endif
// =============================================================================
// @todo (11/02/2019): openmp NOT WORKING: threads still share resources,
// creating data races and producing incorrect results.
// =============================================================================
// #pragma omp parallel for schedule(static, 8)
// =============================================================================
#ifdef USE_BLAS
  const bool kAvailableBLAS = true;
#else
  const bool kAvailableBLAS = false;
#endif

#ifdef SDS_DESIGN
  sw_ctr.start();
#elif !defined(__SYNTHESIS__)
  auto begin_timestep = std::chrono::high_resolution_clock::now();
#endif
  if (use_blas && kAvailableBLAS) {
#ifdef USE_BLAS
    for (int i = 0; i < num_samples; ++i) {
      svd_set(hidden_size, (float)0., c);
      svd_set(hidden_size, (float)0., &out[i * hidden_size]);

      for (int j = 0; j < num_timesteps; ++j) {
        // =======================================================================
        // Current LSTM gates
        // =======================================================================
        // Current us.T @ x: (input_size, n_steps).T @ (input_size, 1)
        svd_cpu_gemv(CblasNoTrans, n_steps, input_size, (float)1., cur_i_u_T, &x[i * kSampleSize + j * input_size], (float)0., cur_i_ux);
        svd_cpu_gemv(CblasNoTrans, n_steps, input_size, (float)1., cur_f_u_T, &x[i * kSampleSize + j * input_size], (float)0., cur_f_ux);
        svd_cpu_gemv(CblasNoTrans, n_steps, input_size, (float)1., cur_c_u_T, &x[i * kSampleSize + j * input_size], (float)0., cur_c_ux);
        svd_cpu_gemv(CblasNoTrans, n_steps, input_size, (float)1., cur_o_u_T, &x[i * kSampleSize + j * input_size], (float)0., cur_o_ux);

        // NOTE: We can skip later bias addition if we memcpy the bias into cur_y,
        // that's because gemv performs: C = alpha * A * B + beta * C, notice
        // that beta is indeed set to 1 here.
        svd_copy(hidden_size, bias_i, i_cur_bias);
        svd_copy(hidden_size, bias_f, f_cur_bias);
        svd_copy(hidden_size, bias_c, c_cur_bias);
        svd_copy(hidden_size, bias_o, o_cur_bias);
        // Current v.T @ ux: (n_steps, hidden_size).T @ (n_steps, 1)
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., cur_i_v_T, cur_i_ux, (float)1., i_cur_bias);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., cur_f_v_T, cur_f_ux, (float)1., f_cur_bias);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., cur_c_v_T, cur_c_ux, (float)1., c_cur_bias);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., cur_o_v_T, cur_o_ux, (float)1., o_cur_bias);

        // =======================================================================
        // Recurrent LSTM gates
        // =======================================================================
        // Recurrent us.T @ h: (hidden_size, n_steps).T @ (hidden_size, 1)
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_i_u_T, &out[i * hidden_size], (float)0., rec_i_uh);
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_f_u_T, &out[i * hidden_size], (float)0., rec_f_uh);
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_c_u_T, &out[i * hidden_size], (float)0., rec_c_uh);
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_o_u_T, &out[i * hidden_size], (float)0., rec_o_uh);

        // Recurrent v.T @ uh: (n_steps, hidden_size).T @ (n_steps, 1)
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_i_v_T, rec_i_uh, (float)0., rec_i_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_f_v_T, rec_f_uh, (float)0., rec_f_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_c_v_T, rec_c_uh, (float)0., rec_c_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_o_v_T, rec_o_uh, (float)0., rec_o_y);
        // =======================================================================
        // Non linearities
        // =======================================================================
        // NOTE: We can skip the bias addition if we previously memcpy-ed
        // the bias into cur_bias, that's because gemv performs: C = alpha * A * B + beta * C
        // and C can be initialized to the bias.
        svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
        svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
        svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
        svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

        svd_hard_sigmoid(hidden_size, i_sum, i_gate);
        svd_hard_sigmoid(hidden_size, f_sum, f_gate);
        svd_hard_sigmoid(hidden_size, o_sum, o_gate);
        svd_tanh(hidden_size, c_sum, c_sum_tanh);
        svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
        svd_mul(hidden_size, c, f_gate, c_rhs);

        svd_add(hidden_size, c_lhs, c_rhs, c);
        svd_tanh(hidden_size, c, c_tanh);
        svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
      }
    }
#endif // end USE_BLAS
  } else {
// #pragma omp parallel for num_threads(8) schedule(dynamic)
    for (int i = 0; i < num_samples; ++i) {
      svd_set(hidden_size, (float)0., c);
      svd_set(hidden_size, (float)0., &out[i * hidden_size]);

      for (int j = 0; j < num_timesteps; ++j) {
        // =======================================================================
        // Current LSTM gates
        // =======================================================================
        // NOTE: in this unbatched version, current and recurrent gates execution
        // is simmetrical, i.e. same transposed matrices logic.
        // =======================================================================
        // us.T @ x
        hls_gemv<float, float>(n_steps, input_size, cur_i_u_T, &x[i * kSampleSize + j * input_size], cur_i_ux);
        hls_gemv<float, float>(n_steps, input_size, cur_f_u_T, &x[i * kSampleSize + j * input_size], cur_f_ux);
        hls_gemv<float, float>(n_steps, input_size, cur_c_u_T, &x[i * kSampleSize + j * input_size], cur_c_ux);
        hls_gemv<float, float>(n_steps, input_size, cur_o_u_T, &x[i * kSampleSize + j * input_size], cur_o_ux);
   
        // v.T @ xus
        hls_gemv<float, float>(hidden_size, n_steps, cur_i_v_T, cur_i_ux, cur_i_y);
        hls_gemv<float, float>(hidden_size, n_steps, cur_f_v_T, cur_f_ux, cur_f_y);
        hls_gemv<float, float>(hidden_size, n_steps, cur_c_v_T, cur_c_ux, cur_c_y);
        hls_gemv<float, float>(hidden_size, n_steps, cur_o_v_T, cur_o_ux, cur_o_y);

        // =======================================================================
        // Recurrent LSTM gates
        // =======================================================================
        // us.T @ h
        hls_gemv<float, float>(n_steps, hidden_size, rec_i_u_T, &out[i * hidden_size], rec_i_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_f_u_T, &out[i * hidden_size], rec_f_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_c_u_T, &out[i * hidden_size], rec_c_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_o_u_T, &out[i * hidden_size], rec_o_uh);
        // v.T @ hus
        hls_gemv<float, float>(hidden_size, n_steps, rec_i_v_T, rec_i_uh, rec_i_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_f_v_T, rec_f_uh, rec_f_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_c_v_T, rec_c_uh, rec_c_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_o_v_T, rec_o_uh, rec_o_y);
        // =======================================================================
        // Non linearities
        // =======================================================================
        // NOTE: We can skip the following bias addition if we previously memcpy
        // the bias into cur_bias, that's because gemv performs: C = alpha * A * B + beta * C
        // and C can be initialized to the bias.
        svd_add(hidden_size, cur_i_y, bias_i, i_cur_bias);
        svd_add(hidden_size, cur_f_y, bias_f, f_cur_bias);
        svd_add(hidden_size, cur_c_y, bias_c, c_cur_bias);
        svd_add(hidden_size, cur_o_y, bias_o, o_cur_bias);

        svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
        svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
        svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
        svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

        svd_hard_sigmoid(hidden_size, i_sum, i_gate);
        svd_hard_sigmoid(hidden_size, f_sum, f_gate);
        svd_hard_sigmoid(hidden_size, o_sum, o_gate);
        svd_tanh(hidden_size, c_sum, c_sum_tanh);
        svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
        svd_mul(hidden_size, c, f_gate, c_rhs);

        svd_add(hidden_size, c_lhs, c_rhs, c);
        svd_tanh(hidden_size, c, c_tanh);
        svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
      }
    }
  }
#ifdef SDS_DESIGN
  sw_ctr.stop();
#elif !defined(__SYNTHESIS__)
  auto end_timestep = std::chrono::high_resolution_clock::now();
  total_time += end_timestep - begin_timestep;
#endif
  // ===========================================================================
  // NOTE: We are NOT taking into account the time it takes to both setup the u @ s
  // matrices and perform their transpositions because these operations can be
  // done "offline", i.e. can be stored in that form already, performance-wise.
  // ===========================================================================
  if (verbose == 1) {
#ifdef SDS_DESIGN
    auto sw_cycles = sw_ctr.avg_cpu_cycles();
    auto sw_freq = sds_clock_frequency();
    std::cout << "Frequency: " << sw_freq << " ticks/second\n";
#ifdef USE_BLAS
    if (use_blas) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
#endif
    std::cout << "Unbatched SVD: Total CPU cycles: " << std::setprecision(12) << sw_cycles << "\n";
#elif !defined(__SYNTHESIS__)
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(total_time).count();
    auto duration_us = duration_ns / 1000.0;
    auto duration_ms = duration_us / 1000.0;
    auto duration_s = duration_ms / 1000.0;
    #ifdef USE_BLAS
    if (use_blas) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
#endif
    std::cout << "Unbatched SVD: Total time: " << duration_ms
              << " ms (" << duration_s << " s)."<< "\n";
#ifdef USE_BLAS
    if (use_blas) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
#endif
    std::cout << "Unbatched SVD: Average time per sample: " << duration_ms / num_samples
              << " ms (" << duration_s / num_samples << " s)."<< "\n";
#ifdef USE_BLAS
    if (use_blas) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
#endif
    std::cout << "Unbatched SVD: Average time per timesteps: " << duration_ms / num_samples / num_timesteps
              << " ms (" << duration_s / num_samples / num_timesteps << " s)."<< "\n";
#endif
  }
  delete[] cur_i_ux;
  delete[] cur_f_ux;
  delete[] cur_c_ux;
  delete[] cur_o_ux;
  delete[] rec_i_uh;
  delete[] rec_f_uh;
  delete[] rec_c_uh;
  delete[] rec_o_uh;
  delete[] cur_i_y;
  delete[] cur_f_y;
  delete[] cur_c_y;
  delete[] cur_o_y;
  delete[] rec_i_y;
  delete[] rec_f_y;
  delete[] rec_c_y;
  delete[] rec_o_y;
  delete[] i_cur_bias;
  delete[] f_cur_bias;
  delete[] c_cur_bias;
  delete[] o_cur_bias;
  delete[] i_sum;
  delete[] f_sum;
  delete[] c_sum;
  delete[] o_sum;
  delete[] i_gate;
  delete[] f_gate;
  delete[] o_gate;
  delete[] c_sum_tanh;
  delete[] c_tanh;
  delete[] c_lhs;
  delete[] c_rhs;
  delete[] c;
  delete[] rec_i_v_T;
  delete[] rec_f_v_T;
  delete[] rec_c_v_T;
  delete[] rec_o_v_T;
  delete[] cur_i_u_T;
  delete[] cur_f_u_T;
  delete[] cur_c_u_T;
  delete[] cur_o_u_T;
  delete[] rec_i_u_T;
  delete[] rec_f_u_T;
  delete[] rec_c_u_T;
  delete[] rec_o_u_T;
  delete[] cur_i_us;
  delete[] cur_f_us;
  delete[] cur_c_us;
  delete[] cur_o_us;
  delete[] rec_i_us;
  delete[] rec_f_us;
  delete[] rec_c_us;
  delete[] rec_o_us;
}

#ifdef __cplusplus
extern "C"
#endif
void SvdModel2LstmFix8(const int verbose,
                       const Fix8D *x,
                       const int num_samples,
                       const int num_timesteps,
                       const int n_steps,
                       const int input_size,
                       const int hidden_size,
                       const Fix8D *cur_i_u,
                       const Fix8D *cur_i_s,
                       const Fix8D *cur_i_v,
                       const Fix8D *cur_f_u,
                       const Fix8D *cur_f_s,
                       const Fix8D *cur_f_v,
                       const Fix8D *cur_c_u,
                       const Fix8D *cur_c_s,
                       const Fix8D *cur_c_v,
                       const Fix8D *cur_o_u,
                       const Fix8D *cur_o_s,
                       const Fix8D *cur_o_v,
                       const Fix8D *rec_i_u,
                       const Fix8D *rec_i_s,
                       const Fix8D *rec_i_v,
                       const Fix8D *rec_f_u,
                       const Fix8D *rec_f_s,
                       const Fix8D *rec_f_v,
                       const Fix8D *rec_c_u,
                       const Fix8D *rec_c_s,
                       const Fix8D *rec_c_v,
                       const Fix8D *rec_o_u,
                       const Fix8D *rec_o_s,
                       const Fix8D *rec_o_v,
                       const Fix8D *bias_i,
                       const Fix8D *bias_f,
                       const Fix8D *bias_c,
                       const Fix8D *bias_o,
                       Fix8D *out) {
  SvdModel2LstmTemplatedLatencyCC<Fix8D, Accum8D, 512>(verbose, x,
                                              num_samples,
                                              num_timesteps,
                                              n_steps,
                                              input_size,
                                              hidden_size,
                                              cur_i_u,
                                              cur_i_s,
                                              cur_i_v,
                                              cur_f_u,
                                              cur_f_s,
                                              cur_f_v,
                                              cur_c_u,
                                              cur_c_s,
                                              cur_c_v,
                                              cur_o_u,
                                              cur_o_s,
                                              cur_o_v,
                                              rec_i_u,
                                              rec_i_s,
                                              rec_i_v,
                                              rec_f_u,
                                              rec_f_s,
                                              rec_f_v,
                                              rec_c_u,
                                              rec_c_s,
                                              rec_c_v,
                                              rec_o_u,
                                              rec_o_s,
                                              rec_o_v,
                                              bias_i,
                                              bias_f,
                                              bias_c,
                                              bias_o,
                                              out);
}

#ifdef __cplusplus
extern "C"
#endif
void SvdModel2LstmFix16(const int verbose,
                        const Fix16D *x,
                        const int num_samples,
                        const int num_timesteps,
                        const int n_steps,
                        const int input_size,
                        const int hidden_size,
                        const Fix16D *cur_i_u,
                        const Fix16D *cur_i_s,
                        const Fix16D *cur_i_v,
                        const Fix16D *cur_f_u,
                        const Fix16D *cur_f_s,
                        const Fix16D *cur_f_v,
                        const Fix16D *cur_c_u,
                        const Fix16D *cur_c_s,
                        const Fix16D *cur_c_v,
                        const Fix16D *cur_o_u,
                        const Fix16D *cur_o_s,
                        const Fix16D *cur_o_v,
                        const Fix16D *rec_i_u,
                        const Fix16D *rec_i_s,
                        const Fix16D *rec_i_v,
                        const Fix16D *rec_f_u,
                        const Fix16D *rec_f_s,
                        const Fix16D *rec_f_v,
                        const Fix16D *rec_c_u,
                        const Fix16D *rec_c_s,
                        const Fix16D *rec_c_v,
                        const Fix16D *rec_o_u,
                        const Fix16D *rec_o_s,
                        const Fix16D *rec_o_v,
                        const Fix16D *bias_i,
                        const Fix16D *bias_f,
                        const Fix16D *bias_c,
                        const Fix16D *bias_o,
                        Fix16D *out) {
  SvdModel2LstmTemplatedLatencyCC<Fix16D, Accum16D, 512>(verbose, x,
                                               num_samples,
                                               num_timesteps,
                                               n_steps,
                                               input_size,
                                               hidden_size,
                                               cur_i_u,
                                               cur_i_s,
                                               cur_i_v,
                                               cur_f_u,
                                               cur_f_s,
                                               cur_f_v,
                                               cur_c_u,
                                               cur_c_s,
                                               cur_c_v,
                                               cur_o_u,
                                               cur_o_s,
                                               cur_o_v,
                                               rec_i_u,
                                               rec_i_s,
                                               rec_i_v,
                                               rec_f_u,
                                               rec_f_s,
                                               rec_f_v,
                                               rec_c_u,
                                               rec_c_s,
                                               rec_c_v,
                                               rec_o_u,
                                               rec_o_s,
                                               rec_o_v,
                                               bias_i,
                                               bias_f,
                                               bias_c,
                                               bias_o,
                                               out);
}

#ifdef __cplusplus
extern "C"
#endif
void SvdModel2LstmHalf(const int verbose,
                        const HalfD *x,
                        const int num_samples,
                        const int num_timesteps,
                        const int n_steps,
                        const int input_size,
                        const int hidden_size,
                        const HalfD *cur_i_u,
                        const HalfD *cur_i_s,
                        const HalfD *cur_i_v,
                        const HalfD *cur_f_u,
                        const HalfD *cur_f_s,
                        const HalfD *cur_f_v,
                        const HalfD *cur_c_u,
                        const HalfD *cur_c_s,
                        const HalfD *cur_c_v,
                        const HalfD *cur_o_u,
                        const HalfD *cur_o_s,
                        const HalfD *cur_o_v,
                        const HalfD *rec_i_u,
                        const HalfD *rec_i_s,
                        const HalfD *rec_i_v,
                        const HalfD *rec_f_u,
                        const HalfD *rec_f_s,
                        const HalfD *rec_f_v,
                        const HalfD *rec_c_u,
                        const HalfD *rec_c_s,
                        const HalfD *rec_c_v,
                        const HalfD *rec_o_u,
                        const HalfD *rec_o_s,
                        const HalfD *rec_o_v,
                        const HalfD *bias_i,
                        const HalfD *bias_f,
                        const HalfD *bias_c,
                        const HalfD *bias_o,
                        HalfD *out) {
  SvdModel2LstmTemplatedLatencyCC<HalfD, AccumHalfD, 256>(verbose, x,
                                               num_samples,
                                               num_timesteps,
                                               n_steps,
                                               input_size,
                                               hidden_size,
                                               cur_i_u,
                                               cur_i_s,
                                               cur_i_v,
                                               cur_f_u,
                                               cur_f_s,
                                               cur_f_v,
                                               cur_c_u,
                                               cur_c_s,
                                               cur_c_v,
                                               cur_o_u,
                                               cur_o_s,
                                               cur_o_v,
                                               rec_i_u,
                                               rec_i_s,
                                               rec_i_v,
                                               rec_f_u,
                                               rec_f_s,
                                               rec_f_v,
                                               rec_c_u,
                                               rec_c_s,
                                               rec_c_v,
                                               rec_o_u,
                                               rec_o_s,
                                               rec_o_v,
                                               bias_i,
                                               bias_f,
                                               bias_c,
                                               bias_o,
                                               out);
}

#ifdef __cplusplus
extern "C"
#endif
void SvdModel2LstmSoftware(const int verbose,
                           const bool use_blas,
                           const int type, // 0:float, 1:fix8, 2:fix16, 3:half16
                           const float *x,
                           const int num_samples,
                           const int num_timesteps,
                           const int n_steps,
                           const int input_size,
                           const int hidden_size,
                           const float *cur_i_u,
                           const float *cur_i_s,
                           const float *cur_i_v,
                           const float *cur_f_u,
                           const float *cur_f_s,
                           const float *cur_f_v,
                           const float *cur_c_u,
                           const float *cur_c_s,
                           const float *cur_c_v,
                           const float *cur_o_u,
                           const float *cur_o_s,
                           const float *cur_o_v,
                           const float *rec_i_u,
                           const float *rec_i_s,
                           const float *rec_i_v,
                           const float *rec_f_u,
                           const float *rec_f_s,
                           const float *rec_f_v,
                           const float *rec_c_u,
                           const float *rec_c_s,
                           const float *rec_c_v,
                           const float *rec_o_u,
                           const float *rec_o_s,
                           const float *rec_o_v,
                           const float *bias_i,
                           const float *bias_f,
                           const float *bias_c,
                           const float *bias_o,
                           float *out) {
  switch(type) {
    case 0:
        // =====================================================================
        // Float32
        // =====================================================================
        if (use_blas) {
          SvdModelSoftwareUnbatched(verbose, use_blas, x, num_samples, num_timesteps,
                                n_steps, input_size, hidden_size, cur_i_u, cur_i_s,
                                cur_i_v, cur_f_u, cur_f_s, cur_f_v, cur_c_u,
                                cur_c_s, cur_c_v, cur_o_u, cur_o_s, cur_o_v,
                                rec_i_u, rec_i_s, rec_i_v, rec_f_u, rec_f_s,
                                rec_f_v, rec_c_u, rec_c_s, rec_c_v, rec_o_u,
                                rec_o_s, rec_o_v, bias_i, bias_f, bias_c,
                                bias_o, out);

          // SvdModelSoftwareBatched(verbose, use_blas, x, num_samples, num_timesteps,
          //                       n_steps, input_size, hidden_size, cur_i_u, cur_i_s,
          //                       cur_i_v, cur_f_u, cur_f_s, cur_f_v, cur_c_u,
          //                       cur_c_s, cur_c_v, cur_o_u, cur_o_s, cur_o_v,
          //                       rec_i_u, rec_i_s, rec_i_v, rec_f_u, rec_f_s,
          //                       rec_f_v, rec_c_u, rec_c_s, rec_c_v, rec_o_u,
          //                       rec_o_s, rec_o_v, bias_i, bias_f, bias_c,
          //                       bias_o, out);
        } else {
          const bool use_eigen = true;
          SvdModelEigenBatched(verbose, use_eigen, x, num_samples, num_timesteps,
                               n_steps, input_size, hidden_size, cur_i_u, cur_i_s,
                               cur_i_v, cur_f_u, cur_f_s, cur_f_v, cur_c_u,
                               cur_c_s, cur_c_v, cur_o_u, cur_o_s, cur_o_v,
                               rec_i_u, rec_i_s, rec_i_v, rec_f_u, rec_f_s,
                               rec_f_v, rec_c_u, rec_c_s, rec_c_v, rec_o_u,
                               rec_o_s, rec_o_v, bias_i, bias_f, bias_c,
                               bias_o, out);
        }
        break;
    case 1: {
        // =====================================================================
        // Fixed point arrays (8 bit)
        // =====================================================================
        Fix8D *x_fix = new Fix8D[num_samples * num_timesteps * input_size];
        Fix8D *cur_i_u_fix = new Fix8D[n_steps * input_size];
        Fix8D *cur_i_s_fix = new Fix8D[n_steps];
        Fix8D *cur_i_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *cur_f_u_fix = new Fix8D[n_steps * input_size];
        Fix8D *cur_f_s_fix = new Fix8D[n_steps];
        Fix8D *cur_f_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *cur_c_u_fix = new Fix8D[n_steps * input_size];
        Fix8D *cur_c_s_fix = new Fix8D[n_steps];
        Fix8D *cur_c_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *cur_o_u_fix = new Fix8D[n_steps * input_size];
        Fix8D *cur_o_s_fix = new Fix8D[n_steps];
        Fix8D *cur_o_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_i_u_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_i_s_fix = new Fix8D[n_steps];
        Fix8D *rec_i_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_f_u_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_f_s_fix = new Fix8D[n_steps];
        Fix8D *rec_f_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_c_u_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_c_s_fix = new Fix8D[n_steps];
        Fix8D *rec_c_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_o_u_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_o_s_fix = new Fix8D[n_steps];
        Fix8D *rec_o_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *bias_i_fix = new Fix8D[hidden_size];
        Fix8D *bias_f_fix = new Fix8D[hidden_size];
        Fix8D *bias_c_fix = new Fix8D[hidden_size];
        Fix8D *bias_o_fix = new Fix8D[hidden_size];
        Fix8D *out_fix = new Fix8D[num_samples * hidden_size];
        // =====================================================================
        // Copy and cast to fixed point
        // =====================================================================
        hls_copy_cast(num_samples * num_timesteps * input_size, x, x_fix);
        hls_copy_cast(n_steps * input_size, cur_i_u, cur_i_u_fix);
        hls_copy_cast(n_steps, cur_i_s, cur_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_i_v, cur_i_v_fix);
        hls_copy_cast(n_steps * input_size, cur_f_u, cur_f_u_fix);
        hls_copy_cast(n_steps, cur_f_s, cur_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_f_v, cur_f_v_fix);
        hls_copy_cast(n_steps * input_size, cur_c_u, cur_c_u_fix);
        hls_copy_cast(n_steps, cur_c_s, cur_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_c_v, cur_c_v_fix);
        hls_copy_cast(n_steps * input_size, cur_o_u, cur_o_u_fix);
        hls_copy_cast(n_steps, cur_o_s, cur_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_o_v, cur_o_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_u, rec_i_u_fix);
        hls_copy_cast(n_steps, rec_i_s, rec_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_v, rec_i_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_u, rec_f_u_fix);
        hls_copy_cast(n_steps, rec_f_s, rec_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_v, rec_f_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_u, rec_c_u_fix);
        hls_copy_cast(n_steps, rec_c_s, rec_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_v, rec_c_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_u, rec_o_u_fix);
        hls_copy_cast(n_steps, rec_o_s, rec_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_v, rec_o_v_fix);
        hls_copy_cast(hidden_size, bias_i, bias_i_fix);
        hls_copy_cast(hidden_size, bias_f, bias_f_fix);
        hls_copy_cast(hidden_size, bias_c, bias_c_fix);
        hls_copy_cast(hidden_size, bias_o, bias_o_fix);
        // =====================================================================
        // Call function
        // =====================================================================
        SvdModel2LstmFix8(verbose, x_fix, num_samples, num_timesteps, n_steps, input_size,
                          hidden_size, cur_i_u_fix, cur_i_s_fix, cur_i_v_fix,
                          cur_f_u_fix, cur_f_s_fix, cur_f_v_fix, cur_c_u_fix,
                          cur_c_s_fix, cur_c_v_fix, cur_o_u_fix, cur_o_s_fix,
                          cur_o_v_fix, rec_i_u_fix, rec_i_s_fix, rec_i_v_fix,
                          rec_f_u_fix, rec_f_s_fix, rec_f_v_fix, rec_c_u_fix,
                          rec_c_s_fix, rec_c_v_fix, rec_o_u_fix, rec_o_s_fix,
                          rec_o_v_fix, bias_i_fix, bias_f_fix, bias_c_fix,
                          bias_o_fix, out_fix);
        // =====================================================================
        // Writeback
        // =====================================================================
        hls_copy_cast(num_samples * hidden_size, out_fix, out);
        // =====================================================================
        // Cleanup
        // =====================================================================
        delete[] x_fix;
        delete[] cur_i_u_fix;
        delete[] cur_i_s_fix;
        delete[] cur_i_v_fix;
        delete[] cur_f_u_fix;
        delete[] cur_f_s_fix;
        delete[] cur_f_v_fix;
        delete[] cur_c_u_fix;
        delete[] cur_c_s_fix;
        delete[] cur_c_v_fix;
        delete[] cur_o_u_fix;
        delete[] cur_o_s_fix;
        delete[] cur_o_v_fix;
        delete[] rec_i_u_fix;
        delete[] rec_i_s_fix;
        delete[] rec_i_v_fix;
        delete[] rec_f_u_fix;
        delete[] rec_f_s_fix;
        delete[] rec_f_v_fix;
        delete[] rec_c_u_fix;
        delete[] rec_c_s_fix;
        delete[] rec_c_v_fix;
        delete[] rec_o_u_fix;
        delete[] rec_o_s_fix;
        delete[] rec_o_v_fix;
        delete[] bias_i_fix;
        delete[] bias_f_fix;
        delete[] bias_c_fix;
        delete[] bias_o_fix;
        delete[] out_fix;
      }
      break;
    case 2: {
        // =====================================================================
        // Fixed point arrays (16 bit)
        // =====================================================================
        Fix16D *x_fix = new Fix16D[num_samples * num_timesteps * input_size];
        Fix16D *cur_i_u_fix = new Fix16D[n_steps * input_size];
        Fix16D *cur_i_s_fix = new Fix16D[n_steps];
        Fix16D *cur_i_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *cur_f_u_fix = new Fix16D[n_steps * input_size];
        Fix16D *cur_f_s_fix = new Fix16D[n_steps];
        Fix16D *cur_f_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *cur_c_u_fix = new Fix16D[n_steps * input_size];
        Fix16D *cur_c_s_fix = new Fix16D[n_steps];
        Fix16D *cur_c_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *cur_o_u_fix = new Fix16D[n_steps * input_size];
        Fix16D *cur_o_s_fix = new Fix16D[n_steps];
        Fix16D *cur_o_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_i_u_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_i_s_fix = new Fix16D[n_steps];
        Fix16D *rec_i_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_f_u_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_f_s_fix = new Fix16D[n_steps];
        Fix16D *rec_f_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_c_u_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_c_s_fix = new Fix16D[n_steps];
        Fix16D *rec_c_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_o_u_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_o_s_fix = new Fix16D[n_steps];
        Fix16D *rec_o_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *bias_i_fix = new Fix16D[hidden_size];
        Fix16D *bias_f_fix = new Fix16D[hidden_size];
        Fix16D *bias_c_fix = new Fix16D[hidden_size];
        Fix16D *bias_o_fix = new Fix16D[hidden_size];
        Fix16D *out_fix = new Fix16D[num_samples * hidden_size];
        // =====================================================================
        // Copy and cast to fixed point
        // =====================================================================
        hls_copy_cast(num_samples * num_timesteps * input_size, x, x_fix);
        hls_copy_cast(n_steps * input_size, cur_i_u, cur_i_u_fix);
        hls_copy_cast(n_steps, cur_i_s, cur_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_i_v, cur_i_v_fix);
        hls_copy_cast(n_steps * input_size, cur_f_u, cur_f_u_fix);
        hls_copy_cast(n_steps, cur_f_s, cur_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_f_v, cur_f_v_fix);
        hls_copy_cast(n_steps * input_size, cur_c_u, cur_c_u_fix);
        hls_copy_cast(n_steps, cur_c_s, cur_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_c_v, cur_c_v_fix);
        hls_copy_cast(n_steps * input_size, cur_o_u, cur_o_u_fix);
        hls_copy_cast(n_steps, cur_o_s, cur_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_o_v, cur_o_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_u, rec_i_u_fix);
        hls_copy_cast(n_steps, rec_i_s, rec_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_v, rec_i_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_u, rec_f_u_fix);
        hls_copy_cast(n_steps, rec_f_s, rec_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_v, rec_f_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_u, rec_c_u_fix);
        hls_copy_cast(n_steps, rec_c_s, rec_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_v, rec_c_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_u, rec_o_u_fix);
        hls_copy_cast(n_steps, rec_o_s, rec_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_v, rec_o_v_fix);
        hls_copy_cast(hidden_size, bias_i, bias_i_fix);
        hls_copy_cast(hidden_size, bias_f, bias_f_fix);
        hls_copy_cast(hidden_size, bias_c, bias_c_fix);
        hls_copy_cast(hidden_size, bias_o, bias_o_fix);
        // =====================================================================
        // Call function
        // =====================================================================
        SvdModel2LstmFix16(verbose, x_fix, num_samples, num_timesteps, n_steps,
                           input_size, hidden_size, cur_i_u_fix, cur_i_s_fix,
                           cur_i_v_fix, cur_f_u_fix, cur_f_s_fix, cur_f_v_fix,
                           cur_c_u_fix, cur_c_s_fix, cur_c_v_fix, cur_o_u_fix,
                           cur_o_s_fix, cur_o_v_fix, rec_i_u_fix, rec_i_s_fix,
                           rec_i_v_fix, rec_f_u_fix, rec_f_s_fix, rec_f_v_fix,
                           rec_c_u_fix, rec_c_s_fix, rec_c_v_fix, rec_o_u_fix,
                           rec_o_s_fix, rec_o_v_fix, bias_i_fix, bias_f_fix,
                           bias_c_fix, bias_o_fix, out_fix);
        // =====================================================================
        // Writeback
        // =====================================================================
        hls_copy_cast(num_samples * hidden_size, out_fix, out);
        // =====================================================================
        // Cleanup
        // =====================================================================
        delete[] x_fix;
        delete[] cur_i_u_fix;
        delete[] cur_i_s_fix;
        delete[] cur_i_v_fix;
        delete[] cur_f_u_fix;
        delete[] cur_f_s_fix;
        delete[] cur_f_v_fix;
        delete[] cur_c_u_fix;
        delete[] cur_c_s_fix;
        delete[] cur_c_v_fix;
        delete[] cur_o_u_fix;
        delete[] cur_o_s_fix;
        delete[] cur_o_v_fix;
        delete[] rec_i_u_fix;
        delete[] rec_i_s_fix;
        delete[] rec_i_v_fix;
        delete[] rec_f_u_fix;
        delete[] rec_f_s_fix;
        delete[] rec_f_v_fix;
        delete[] rec_c_u_fix;
        delete[] rec_c_s_fix;
        delete[] rec_c_v_fix;
        delete[] rec_o_u_fix;
        delete[] rec_o_s_fix;
        delete[] rec_o_v_fix;
        delete[] bias_i_fix;
        delete[] bias_f_fix;
        delete[] bias_c_fix;
        delete[] bias_o_fix;
        delete[] out_fix;
      }
      break;
    case 3: {
        // =====================================================================
        // Half floating point (16 bit)
        // =====================================================================
        HalfD *x_fix = new HalfD[num_samples * num_timesteps * input_size];
        HalfD *cur_i_u_fix = new HalfD[n_steps * input_size];
        HalfD *cur_i_s_fix = new HalfD[n_steps];
        HalfD *cur_i_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *cur_f_u_fix = new HalfD[n_steps * input_size];
        HalfD *cur_f_s_fix = new HalfD[n_steps];
        HalfD *cur_f_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *cur_c_u_fix = new HalfD[n_steps * input_size];
        HalfD *cur_c_s_fix = new HalfD[n_steps];
        HalfD *cur_c_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *cur_o_u_fix = new HalfD[n_steps * input_size];
        HalfD *cur_o_s_fix = new HalfD[n_steps];
        HalfD *cur_o_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_i_u_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_i_s_fix = new HalfD[n_steps];
        HalfD *rec_i_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_f_u_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_f_s_fix = new HalfD[n_steps];
        HalfD *rec_f_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_c_u_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_c_s_fix = new HalfD[n_steps];
        HalfD *rec_c_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_o_u_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_o_s_fix = new HalfD[n_steps];
        HalfD *rec_o_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *bias_i_fix = new HalfD[hidden_size];
        HalfD *bias_f_fix = new HalfD[hidden_size];
        HalfD *bias_c_fix = new HalfD[hidden_size];
        HalfD *bias_o_fix = new HalfD[hidden_size];
        HalfD *out_fix = new HalfD[num_samples * hidden_size];
        // =====================================================================
        // Copy and cast to fixed point
        // =====================================================================
        // std::cout << "Starting converting float to half\n";
        hls_copy_cast(num_samples * num_timesteps * input_size, x, x_fix);
        hls_copy_cast(n_steps * input_size, cur_i_u, cur_i_u_fix);
        hls_copy_cast(n_steps, cur_i_s, cur_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_i_v, cur_i_v_fix);
        hls_copy_cast(n_steps * input_size, cur_f_u, cur_f_u_fix);
        hls_copy_cast(n_steps, cur_f_s, cur_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_f_v, cur_f_v_fix);
        hls_copy_cast(n_steps * input_size, cur_c_u, cur_c_u_fix);
        hls_copy_cast(n_steps, cur_c_s, cur_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_c_v, cur_c_v_fix);
        hls_copy_cast(n_steps * input_size, cur_o_u, cur_o_u_fix);
        hls_copy_cast(n_steps, cur_o_s, cur_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_o_v, cur_o_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_u, rec_i_u_fix);
        hls_copy_cast(n_steps, rec_i_s, rec_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_v, rec_i_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_u, rec_f_u_fix);
        hls_copy_cast(n_steps, rec_f_s, rec_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_v, rec_f_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_u, rec_c_u_fix);
        hls_copy_cast(n_steps, rec_c_s, rec_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_v, rec_c_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_u, rec_o_u_fix);
        hls_copy_cast(n_steps, rec_o_s, rec_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_v, rec_o_v_fix);
        hls_copy_cast(hidden_size, bias_i, bias_i_fix);
        hls_copy_cast(hidden_size, bias_f, bias_f_fix);
        hls_copy_cast(hidden_size, bias_c, bias_c_fix);
        hls_copy_cast(hidden_size, bias_o, bias_o_fix);
        // =====================================================================
        // Call function
        // =====================================================================
        // std::cout << "Starting SvdModel2LstmHalf\n";
        SvdModel2LstmHalf(verbose, x_fix, num_samples, num_timesteps, n_steps,
                           input_size, hidden_size, cur_i_u_fix, cur_i_s_fix,
                           cur_i_v_fix, cur_f_u_fix, cur_f_s_fix, cur_f_v_fix,
                           cur_c_u_fix, cur_c_s_fix, cur_c_v_fix, cur_o_u_fix,
                           cur_o_s_fix, cur_o_v_fix, rec_i_u_fix, rec_i_s_fix,
                           rec_i_v_fix, rec_f_u_fix, rec_f_s_fix, rec_f_v_fix,
                           rec_c_u_fix, rec_c_s_fix, rec_c_v_fix, rec_o_u_fix,
                           rec_o_s_fix, rec_o_v_fix, bias_i_fix, bias_f_fix,
                           bias_c_fix, bias_o_fix, out_fix);
        // =====================================================================
        // Writeback
        // =====================================================================
        hls_copy_cast(num_samples * hidden_size, out_fix, out);
        // =====================================================================
        // Cleanup
        // =====================================================================
        delete[] x_fix;
        delete[] cur_i_u_fix;
        delete[] cur_i_s_fix;
        delete[] cur_i_v_fix;
        delete[] cur_f_u_fix;
        delete[] cur_f_s_fix;
        delete[] cur_f_v_fix;
        delete[] cur_c_u_fix;
        delete[] cur_c_s_fix;
        delete[] cur_c_v_fix;
        delete[] cur_o_u_fix;
        delete[] cur_o_s_fix;
        delete[] cur_o_v_fix;
        delete[] rec_i_u_fix;
        delete[] rec_i_s_fix;
        delete[] rec_i_v_fix;
        delete[] rec_f_u_fix;
        delete[] rec_f_s_fix;
        delete[] rec_f_v_fix;
        delete[] rec_c_u_fix;
        delete[] rec_c_s_fix;
        delete[] rec_c_v_fix;
        delete[] rec_o_u_fix;
        delete[] rec_o_s_fix;
        delete[] rec_o_v_fix;
        delete[] bias_i_fix;
        delete[] bias_f_fix;
        delete[] bias_c_fix;
        delete[] bias_o_fix;
        delete[] out_fix;
      }
      break;
    default: {
        // =====================================================================
        // Float32
        // =====================================================================
        SvdModelSoftwareUnbatched(verbose, use_blas, x, num_samples, num_timesteps,
                              n_steps, input_size, hidden_size, cur_i_u, cur_i_s,
                              cur_i_v, cur_f_u, cur_f_s, cur_f_v, cur_c_u,
                              cur_c_s, cur_c_v, cur_o_u, cur_o_s, cur_o_v,
                              rec_i_u, rec_i_s, rec_i_v, rec_f_u, rec_f_s,
                              rec_f_v, rec_c_u, rec_c_s, rec_c_v, rec_o_u,
                              rec_o_s, rec_o_v, bias_i, bias_f, bias_c,
                              bias_o, out);
      }
      break;
  }
}

#ifdef EIGEN_DESIGN
void FillSparseMatrix(const int m, const int n, const float *dense_mat, SpMatD &sparse_mat) {
  std::vector<TripletD> triplet_list;

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (dense_mat[i * n + j] != 0) {
        triplet_list.push_back(TripletD(i, j, dense_mat[i * n + j]));
      }
    }
  }
  sparse_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
}
#endif

#ifdef __cplusplus
extern "C"
#endif
void SvdModelEigenBatched(const int verbose,
                          const bool use_eigen,
                          const float *x,
                          const int num_samples,
                          const int num_timesteps,
                          const int n_steps,
                          const int input_size,
                          const int hidden_size,
                          const float *cur_i_u,
                          const float *cur_i_s,
                          const float *cur_i_v,
                          const float *cur_f_u,
                          const float *cur_f_s,
                          const float *cur_f_v,
                          const float *cur_c_u,
                          const float *cur_c_s,
                          const float *cur_c_v,
                          const float *cur_o_u,
                          const float *cur_o_s,
                          const float *cur_o_v,
                          const float *rec_i_u,
                          const float *rec_i_s,
                          const float *rec_i_v,
                          const float *rec_f_u,
                          const float *rec_f_s,
                          const float *rec_f_v,
                          const float *rec_c_u,
                          const float *rec_c_s,
                          const float *rec_c_v,
                          const float *rec_o_u,
                          const float *rec_o_s,
                          const float *rec_o_v,
                          const float *bias_i,
                          const float *bias_f,
                          const float *bias_c,
                          const float *bias_o,
                          float *out) {

#ifdef EIGEN_DESIGN
  const bool kAvailableEigen = true;
  if (verbose && use_eigen) {
    std::cout << "[INFO] Using Eigen library with " << Eigen::nbThreads() << " threads.\n";
  }
#else
  const bool kAvailableEigen = false;
#endif
  // ===========================================================================
  // This C++ function is implementing the following python snippet:
  // (Note the extensive use of matrix multiplications)
  // 
  //   m = 6 # input size
  //   n = 8 # output size, i.e. LSTM hidden size (in Keras)
  //   b = 4 # batch size (or timesteps)
  //   k = 2 # num iterations, i.e. n_steps
  //   x = np.random.randn(m, b)
  //   u = np.random.randn(m, k)
  //   v = np.random.randn(n, k)
  //   s = np.random.randn(k)
  //   s2 = np.random.randn(k)
  //   
  //   s_mat = np.repeat(s, m).reshape(k, m).T
  //   us = u * s_mat
  //   ux = x.T @ us
  //   y_batched = ux @ v.T
  // ===========================================================================
  // ===========================================================================
  // Define: x.T @ US Matrixes
  // ===========================================================================
  // Current x
  float *cur_i_ux = new float[num_timesteps * n_steps];
  float *cur_f_ux = new float[num_timesteps * n_steps];
  float *cur_c_ux = new float[num_timesteps * n_steps];
  float *cur_o_ux = new float[num_timesteps * n_steps];
  // Recurrent x
  float *rec_i_uh = new float[n_steps];
  float *rec_f_uh = new float[n_steps];
  float *rec_c_uh = new float[n_steps];
  float *rec_o_uh = new float[n_steps];

  // Current y
  float *cur_i_y;
  float *cur_f_y;
  float *cur_c_y;
  float *cur_o_y;
  // Recurrent y
  float *rec_i_y;
  float *rec_f_y;
  float *rec_c_y;
  float *rec_o_y;
#ifndef EIGEN_DESIGN
  if (!(use_eigen && kAvailableEigen)) {
    // Current y
    cur_i_y = new float[num_timesteps * hidden_size];
    cur_f_y = new float[num_timesteps * hidden_size];
    cur_c_y = new float[num_timesteps * hidden_size];
    cur_o_y = new float[num_timesteps * hidden_size];
    // Recurrent y
    rec_i_y = new float[hidden_size];
    rec_f_y = new float[hidden_size];
    rec_c_y = new float[hidden_size];
    rec_o_y = new float[hidden_size];
  }
#endif
  // ===========================================================================
  // NOTE: Broadcasting bias vector into a matrix for exploiting C initialization
  // in gemm function (C = A * B + C) produces no significant gain in terms of
  // execution time and makes the code harder to read...
  // ===========================================================================
  // Output y + bias
  float *i_cur_bias = new float[num_timesteps * hidden_size];
  float *f_cur_bias = new float[num_timesteps * hidden_size];
  float *c_cur_bias = new float[num_timesteps * hidden_size];
  float *o_cur_bias = new float[num_timesteps * hidden_size];

  float *i_sum = new float[hidden_size];
  float *f_sum = new float[hidden_size];
  float *c_sum = new float[hidden_size];
  float *o_sum = new float[hidden_size];

  float *i_gate = new float[hidden_size];
  float *f_gate = new float[hidden_size];
  float *o_gate = new float[hidden_size];
  float *c_sum_tanh = new float[hidden_size];
  float *c_tanh = new float[hidden_size];
  float *c_lhs = new float[hidden_size];
  float *c_rhs = new float[hidden_size];
  float *c = new float[hidden_size];

  // Recurrent v Transposed
  float *rec_i_v_T = new float[hidden_size * n_steps];
  float *rec_f_v_T = new float[hidden_size * n_steps];
  float *rec_c_v_T = new float[hidden_size * n_steps];
  float *rec_o_v_T = new float[hidden_size * n_steps];
  // ===========================================================================
  // Compute:
  // - S * U Matrix
  // ===========================================================================
  // Current us1
  float *cur_i_u_T = new float[n_steps * input_size];
  float *cur_f_u_T = new float[n_steps * input_size];
  float *cur_c_u_T = new float[n_steps * input_size];
  float *cur_o_u_T = new float[n_steps * input_size];
  // Recurrent us
  float *rec_i_u_T = new float[n_steps * hidden_size];
  float *rec_f_u_T = new float[n_steps * hidden_size];
  float *rec_c_u_T = new float[n_steps * hidden_size];
  float *rec_o_u_T = new float[n_steps * hidden_size];
  // Current us1
  float *cur_i_us = new float[n_steps * input_size];
  float *cur_f_us = new float[n_steps * input_size];
  float *cur_c_us = new float[n_steps * input_size];
  float *cur_o_us = new float[n_steps * input_size];
  // Recurrent us
  float *rec_i_us = new float[n_steps * hidden_size];
  float *rec_f_us = new float[n_steps * hidden_size];
  float *rec_c_us = new float[n_steps * hidden_size];
  float *rec_o_us = new float[n_steps * hidden_size];

  // ===========================================================================
  // NOTE: We need to 'transpose' u in order to generate the us matrix. This is
  // because the multiplication svd_mul() operates row-wise.
  // ===========================================================================
  // BEFORE TRANSPOSE: s.shape = (n_steps)
  // BEFORE TRANSPOSE: u.shape = (n_steps, input_size)
  // BEFORE TRANSPOSE: us.shape = (n_steps, input_size)
  svd_transpose(n_steps, input_size, cur_i_u, cur_i_u_T);
  svd_transpose(n_steps, input_size, cur_f_u, cur_f_u_T);
  svd_transpose(n_steps, input_size, cur_c_u, cur_c_u_T);
  svd_transpose(n_steps, input_size, cur_o_u, cur_o_u_T);
  svd_transpose(n_steps, hidden_size, rec_i_u, rec_i_u_T);
  svd_transpose(n_steps, hidden_size, rec_f_u, rec_f_u_T);
  svd_transpose(n_steps, hidden_size, rec_c_u, rec_c_u_T);
  svd_transpose(n_steps, hidden_size, rec_o_u, rec_o_u_T);
  for (int i = 0; i < input_size; ++i) {
    svd_mul(n_steps, &cur_i_u_T[i*n_steps], cur_i_s, &cur_i_us[i*n_steps]);
    svd_mul(n_steps, &cur_f_u_T[i*n_steps], cur_f_s, &cur_f_us[i*n_steps]);
    svd_mul(n_steps, &cur_c_u_T[i*n_steps], cur_c_s, &cur_c_us[i*n_steps]);
    svd_mul(n_steps, &cur_o_u_T[i*n_steps], cur_o_s, &cur_o_us[i*n_steps]);
  }
  for (int i = 0; i < hidden_size; ++i) {
    svd_mul(n_steps, &rec_i_u_T[i*n_steps], rec_i_s, &rec_i_us[i*n_steps]);
    svd_mul(n_steps, &rec_f_u_T[i*n_steps], rec_f_s, &rec_f_us[i*n_steps]);
    svd_mul(n_steps, &rec_c_u_T[i*n_steps], rec_c_s, &rec_c_us[i*n_steps]);
    svd_mul(n_steps, &rec_o_u_T[i*n_steps], rec_o_s, &rec_o_us[i*n_steps]);
  }
  // ===========================================================================
  // Transpose recurrent v and recurrent u vectors.
  // ===========================================================================
  // From (hidden_size, n_steps) to (n_steps, hidden_size)
  svd_transpose(hidden_size, n_steps, rec_i_us, rec_i_u_T);
  svd_transpose(hidden_size, n_steps, rec_f_us, rec_f_u_T);
  svd_transpose(hidden_size, n_steps, rec_c_us, rec_c_u_T);
  svd_transpose(hidden_size, n_steps, rec_o_us, rec_o_u_T);
  // From (n_steps, hidden_size) to (hidden_size, n_steps)
  svd_transpose(n_steps, hidden_size, rec_i_v, rec_i_v_T);
  svd_transpose(n_steps, hidden_size, rec_f_v, rec_f_v_T);
  svd_transpose(n_steps, hidden_size, rec_c_v, rec_c_v_T);
  svd_transpose(n_steps, hidden_size, rec_o_v, rec_o_v_T);

#ifdef EIGEN_DESIGN
  std::vector<MatD, Eigen::aligned_allocator<MatD> > x_dense;
  for (int i = 0; i < num_samples; ++i) {
    x_dense.push_back(Eigen::Map<MatD>(const_cast<float*>(&x[i * num_timesteps * input_size]), num_timesteps, input_size));
  }

  // Define the sparse matrixes.
  SpMatD cur_i_us_sp(input_size, n_steps);
  SpMatD cur_f_us_sp(input_size, n_steps);
  SpMatD cur_c_us_sp(input_size, n_steps);
  SpMatD cur_o_us_sp(input_size, n_steps);

  FillSparseMatrix(input_size, n_steps, cur_i_us, cur_i_us_sp);
  FillSparseMatrix(input_size, n_steps, cur_f_us, cur_f_us_sp);
  FillSparseMatrix(input_size, n_steps, cur_c_us, cur_c_us_sp);
  FillSparseMatrix(input_size, n_steps, cur_o_us, cur_o_us_sp);

  SpMatD cur_i_v_sp(n_steps, hidden_size);
  SpMatD cur_f_v_sp(n_steps, hidden_size);
  SpMatD cur_c_v_sp(n_steps, hidden_size);
  SpMatD cur_o_v_sp(n_steps, hidden_size);

  FillSparseMatrix(n_steps, hidden_size, cur_i_v, cur_i_v_sp);
  FillSparseMatrix(n_steps, hidden_size, cur_f_v, cur_f_v_sp);
  FillSparseMatrix(n_steps, hidden_size, cur_c_v, cur_c_v_sp);
  FillSparseMatrix(n_steps, hidden_size, cur_o_v, cur_o_v_sp);

  SpMatD rec_i_us_sp(n_steps, hidden_size);
  SpMatD rec_f_us_sp(n_steps, hidden_size);
  SpMatD rec_c_us_sp(n_steps, hidden_size);
  SpMatD rec_o_us_sp(n_steps, hidden_size);

  FillSparseMatrix(n_steps, hidden_size, rec_i_us, rec_i_us_sp);
  FillSparseMatrix(n_steps, hidden_size, rec_f_us, rec_f_us_sp);
  FillSparseMatrix(n_steps, hidden_size, rec_c_us, rec_c_us_sp);
  FillSparseMatrix(n_steps, hidden_size, rec_o_us, rec_o_us_sp);

  SpMatD rec_i_u_T_sp(n_steps, hidden_size);
  SpMatD rec_f_u_T_sp(n_steps, hidden_size);
  SpMatD rec_c_u_T_sp(n_steps, hidden_size);
  SpMatD rec_o_u_T_sp(n_steps, hidden_size);

  SpMatD rec_i_v_T_sp(hidden_size, n_steps);
  SpMatD rec_f_v_T_sp(hidden_size, n_steps);
  SpMatD rec_c_v_T_sp(hidden_size, n_steps);
  SpMatD rec_o_v_T_sp(hidden_size, n_steps);

  FillSparseMatrix(n_steps, hidden_size, rec_i_u_T, rec_i_u_T_sp);
  FillSparseMatrix(n_steps, hidden_size, rec_f_u_T, rec_f_u_T_sp);
  FillSparseMatrix(n_steps, hidden_size, rec_c_u_T, rec_c_u_T_sp);
  FillSparseMatrix(n_steps, hidden_size, rec_o_u_T, rec_o_u_T_sp);

  FillSparseMatrix(hidden_size, n_steps, rec_i_v_T, rec_i_v_T_sp);
  FillSparseMatrix(hidden_size, n_steps, rec_f_v_T, rec_f_v_T_sp);
  FillSparseMatrix(hidden_size, n_steps, rec_c_v_T, rec_c_v_T_sp);
  FillSparseMatrix(hidden_size, n_steps, rec_o_v_T, rec_o_v_T_sp);

  MatD cur_i_ux_dense = MatD(num_timesteps, n_steps);
  MatD cur_f_ux_dense = MatD(num_timesteps, n_steps);
  MatD cur_c_ux_dense = MatD(num_timesteps, n_steps);
  MatD cur_o_ux_dense = MatD(num_timesteps, n_steps);

  MatD cur_i_y_dense = MatD(n_steps, hidden_size);
  MatD cur_f_y_dense = MatD(n_steps, hidden_size);
  MatD cur_c_y_dense = MatD(n_steps, hidden_size);
  MatD cur_o_y_dense = MatD(n_steps, hidden_size);

  MatD out_dense = MatD(hidden_size, 1);

  MatD rec_i_uh_dense = MatD(hidden_size, 1);
  MatD rec_f_uh_dense = MatD(hidden_size, 1);
  MatD rec_c_uh_dense = MatD(hidden_size, 1);
  MatD rec_o_uh_dense = MatD(hidden_size, 1);

  MatD rec_i_y_dense = MatD(hidden_size, 1);
  MatD rec_f_y_dense = MatD(hidden_size, 1);
  MatD rec_c_y_dense = MatD(hidden_size, 1);
  MatD rec_o_y_dense = MatD(hidden_size, 1);
#endif

#ifdef SDS_DESIGN
  perf_counter sw_ctr;
  perf_counter gemm_ctr;
  perf_counter gemv_ctr;
  sw_ctr.start();
#elif !defined(__SYNTHESIS__)
  auto begin = std::chrono::high_resolution_clock::now();
#endif

  if (use_eigen && kAvailableEigen) {
#ifdef EIGEN_DESIGN
    // =========================================================================
    // @todo (11/02/2019): openmp NOT WORKING: threads still share resources,
    // creating data races and producing incorrect results.
    // =========================================================================
    // #pragma omp parallel for schedule(static, 8)
    // =========================================================================
// #pragma omp parallel for num_threads(8) schedule(dynamic)
    for (int i = 0; i < num_samples; ++i) {
#ifdef SDS_DESIGN
      gemm_ctr.start();
#endif
      // Current x @ us: (num_timesteps, input_size) @ (input_size, n_steps)
      cur_i_ux_dense = x_dense[i] * cur_i_us_sp;
      cur_f_ux_dense = x_dense[i] * cur_f_us_sp;
      cur_c_ux_dense = x_dense[i] * cur_c_us_sp;
      cur_o_ux_dense = x_dense[i] * cur_o_us_sp;

      // Current ux @ v: (num_timesteps, n_steps) @ (n_steps, hidden_size)
      cur_i_y_dense = cur_i_ux_dense * cur_i_v_sp;
      cur_f_y_dense = cur_f_ux_dense * cur_f_v_sp;
      cur_c_y_dense = cur_c_ux_dense * cur_c_v_sp;
      cur_o_y_dense = cur_o_ux_dense * cur_o_v_sp;

      cur_i_y = cur_i_y_dense.data();
      cur_f_y = cur_f_y_dense.data();
      cur_c_y = cur_c_y_dense.data();
      cur_o_y = cur_o_y_dense.data();

#ifdef SDS_DESIGN
      gemm_ctr.stop();
#endif

      svd_set(hidden_size, (float)0., c);

      for (int j = 0; j < num_timesteps; ++j) {
        out_dense = Eigen::Map<MatD>(&out[i * hidden_size], hidden_size, 1);
#ifdef SDS_DESIGN
        gemv_ctr.start();
#endif
        // Recurrent us.T @ h: (hidden_size, n_steps).T @ (hidden_size, 1)
        rec_i_uh_dense = rec_i_u_T_sp * out_dense;
        rec_f_uh_dense = rec_f_u_T_sp * out_dense;
        rec_c_uh_dense = rec_c_u_T_sp * out_dense;
        rec_o_uh_dense = rec_o_u_T_sp * out_dense;

        // Recurrent v.T @ uh: (n_steps, hidden_size).T @ (n_steps, 1)
        rec_i_y_dense = rec_i_v_T_sp * rec_i_uh_dense;
        rec_f_y_dense = rec_f_v_T_sp * rec_f_uh_dense;
        rec_c_y_dense = rec_c_v_T_sp * rec_c_uh_dense;
        rec_o_y_dense = rec_o_v_T_sp * rec_o_uh_dense;

        rec_i_y = rec_i_y_dense.data();
        rec_f_y = rec_f_y_dense.data();
        rec_c_y = rec_c_y_dense.data();
        rec_o_y = rec_o_y_dense.data();

#ifdef SDS_DESIGN
        gemv_ctr.stop();
#endif
        // =======================================================================
        // Non Linearities
        // =======================================================================
        svd_add(hidden_size, &cur_i_y[j * hidden_size], bias_i, i_cur_bias);
        svd_add(hidden_size, &cur_f_y[j * hidden_size], bias_f, f_cur_bias);
        svd_add(hidden_size, &cur_c_y[j * hidden_size], bias_c, c_cur_bias);
        svd_add(hidden_size, &cur_o_y[j * hidden_size], bias_o, o_cur_bias);

        svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
        svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
        svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
        svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

        svd_hard_sigmoid(hidden_size, i_sum, i_gate);
        svd_hard_sigmoid(hidden_size, f_sum, f_gate);
        svd_hard_sigmoid(hidden_size, o_sum, o_gate);
        svd_tanh(hidden_size, c_sum, c_sum_tanh);
        svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
        svd_mul(hidden_size, c, f_gate, c_rhs);

        svd_add(hidden_size, c_lhs, c_rhs, c);
        svd_tanh(hidden_size, c, c_tanh);
        svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
      }
    }
#endif // end EIGEN_DESIGN
  } else {
    // =========================================================================
    // @todo (11/02/2019): openmp NOT WORKING: threads still share resources,
    // creating data races and producing incorrect results.
    // =========================================================================
    // #pragma omp parallel for schedule(static, 8)
    // =========================================================================
// #pragma omp parallel for num_threads(8) schedule(dynamic)
    for (int i = 0; i < num_samples; ++i) {
#ifdef SDS_DESIGN
      gemm_ctr.start();
#endif
      hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_i_us, cur_i_ux);
      hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_f_us, cur_f_ux);
      hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_c_us, cur_c_ux);
      hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_o_us, cur_o_ux);

      hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_i_ux, cur_i_v, cur_i_y);
      hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_f_ux, cur_f_v, cur_f_y);
      hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_c_ux, cur_c_v, cur_c_y);
      hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_o_ux, cur_o_v, cur_o_y);

#ifdef SDS_DESIGN
      gemm_ctr.stop();
#endif
      svd_set(hidden_size, (float)0., &out[i * hidden_size]);
      svd_set(hidden_size, (float)0., c);

      for (int j = 0; j < num_timesteps; ++j) {
#ifdef SDS_DESIGN
        gemv_ctr.start();
#endif
        hls_gemv<float, float>(n_steps, hidden_size, rec_i_u_T, &out[i*hidden_size], rec_i_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_f_u_T, &out[i*hidden_size], rec_f_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_c_u_T, &out[i*hidden_size], rec_c_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_o_u_T, &out[i*hidden_size], rec_o_uh);

        hls_gemv<float, float>(hidden_size, n_steps, rec_i_v_T, rec_i_uh, rec_i_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_f_v_T, rec_f_uh, rec_f_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_c_v_T, rec_c_uh, rec_c_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_o_v_T, rec_o_uh, rec_o_y);
#ifdef SDS_DESIGN
        gemv_ctr.stop();
#endif
        // =======================================================================
        // Non Linearities
        // =======================================================================
        svd_add(hidden_size, &cur_i_y[j * hidden_size], bias_i, i_cur_bias);
        svd_add(hidden_size, &cur_f_y[j * hidden_size], bias_f, f_cur_bias);
        svd_add(hidden_size, &cur_c_y[j * hidden_size], bias_c, c_cur_bias);
        svd_add(hidden_size, &cur_o_y[j * hidden_size], bias_o, o_cur_bias);

        svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
        svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
        svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
        svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

        svd_hard_sigmoid(hidden_size, i_sum, i_gate);
        svd_hard_sigmoid(hidden_size, f_sum, f_gate);
        svd_hard_sigmoid(hidden_size, o_sum, o_gate);
        svd_tanh(hidden_size, c_sum, c_sum_tanh);
        svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
        svd_mul(hidden_size, c, f_gate, c_rhs);

        svd_add(hidden_size, c_lhs, c_rhs, c);
        svd_tanh(hidden_size, c, c_tanh);
        svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
      }
    }
  }
  // ===========================================================================
  // NOTE: We are NOT taking into account the time it takes to both setup the u @ s
  // matrices and perform their transpositions because these operations can be
  // done "offline", i.e. can be stored in that form already, performance-wise.
  // ===========================================================================
#ifdef SDS_DESIGN
  sw_ctr.stop();
#elif !defined(__SYNTHESIS__)
  auto end = std::chrono::high_resolution_clock::now();
#endif

  if (verbose == 1) {
#ifdef SDS_DESIGN
    auto sw_cycles = sw_ctr.avg_cpu_cycles();
    auto sw_freq = sds_clock_frequency();
    std::cout << "Frequency: " << sw_freq << " ticks/second\n";
    if (use_eigen && kAvailableEigen) {
      std::cout << "[Eigen version] ";
    } else {
      std::cout << "[no Eigen] ";
    }
    std::cout << "Batched SVD: Total CPU cycles: " << std::setprecision(12)
              << sw_cycles << std::setprecision(6) << "\n";
    if (use_eigen && kAvailableEigen) {
      std::cout << "[Eigen version] ";
    } else {
      std::cout << "[no Eigen] ";
    }
    std::cout << "Batched SVD: Average CPU cycles per sample: "
              << sw_cycles / num_samples << "\n";
    std::cout << "Batched SVD: Average CPU cycles per timesteps: "
              << sw_cycles / num_samples / num_timesteps << "\n";


    auto gemm_cycles = gemm_ctr.avg_cpu_cycles();
    if (use_eigen && kAvailableEigen) {
      std::cout << "[Eigen version] ";
    } else {
      std::cout << "[no Eigen] ";
    }
    std::cout << "Batched SVD-GEMM: Total CPU cycles: " << std::setprecision(12)
              << gemm_cycles << std::setprecision(6) << "\n";
    if (use_eigen && kAvailableEigen) {
      std::cout << "[Eigen version] ";
    } else {
      std::cout << "[no Eigen] ";
    }
    std::cout << "Batched SVD-GEMM: Average CPU cycles per sample: "
              << gemm_cycles / num_samples << "\n";
    if (use_eigen && kAvailableEigen) {
      std::cout << "[Eigen version] ";
    } else {
      std::cout << "[no Eigen] ";
    }
    std::cout << "Batched SVD-GEMM: Average CPU cycles per timesteps: "
              << gemm_cycles / num_samples / num_timesteps << "\n";

    auto gemv_cycles = gemv_ctr.avg_cpu_cycles();
    if (use_eigen && kAvailableEigen) {
      std::cout << "[Eigen version] ";
    } else {
      std::cout << "[no Eigen] ";
    }
    std::cout << "Batched SVD-GEMV: Total CPU cycles: " << std::setprecision(12)
              << gemv_cycles << std::setprecision(6) << "\n";
    if (use_eigen && kAvailableEigen) {
      std::cout << "[Eigen version] ";
    } else {
      std::cout << "[no Eigen] ";
    }
    std::cout << "Batched SVD-GEMV: Average CPU cycles per sample: "
              << gemv_cycles / num_samples << "\n";
    if (use_eigen && kAvailableEigen) {
      std::cout << "[Eigen version] ";
    } else {
      std::cout << "[no Eigen] ";
    }
    std::cout << "Batched SVD-GEMV: Average CPU cycles per timesteps: "
              << gemv_cycles / num_samples / num_timesteps << "\n";

#elif !defined(__SYNTHESIS__)
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - 
      begin).count();
    auto duration_us = duration_ns / 1000.0;
    auto duration_ms = duration_us / 1000.0;
    auto duration_s = duration_ms / 1000.0;
    if (use_eigen && kAvailableEigen) {
      std::cout << "[Eigen version] ";
    } else {
      std::cout << "[no Eigen] ";
    }
    std::cout << "Batched SVD: Total time: " << duration_ms
              << " ms (" << duration_s << " s)."<< "\n";
    if (use_eigen && kAvailableEigen) {
      std::cout << "[Eigen version] ";
    } else {
      std::cout << "[no Eigen] ";
    }
    std::cout << "Batched SVD: Average time per sample: " << duration_ms / num_samples
              << " ms (" << duration_s / num_samples << " s)."<< "\n";
    if (use_eigen && kAvailableEigen) {
      std::cout << "[Eigen version] ";
    } else {
      std::cout << "[no Eigen] ";
    }
    std::cout << "Batched SVD: Average time per timesteps: " << duration_ms / num_samples / num_timesteps
              << " ms (" << duration_s / num_samples / num_timesteps << " s)."<< "\n";
#endif
  }

  delete[] cur_i_ux;
  delete[] cur_f_ux;
  delete[] cur_c_ux;
  delete[] cur_o_ux;
  delete[] rec_i_uh;
  delete[] rec_f_uh;
  delete[] rec_c_uh;
  delete[] rec_o_uh;

  if (!(use_eigen && kAvailableEigen)) {
    // NOTE: When using Eigen, these pointers are converted to temporary matrix
    // objects.
#ifdef EIGEN_DESIGN
    std::cout << "[EIGEN] Freeing cur_i_y\n";
    delete[] cur_i_y;
    delete[] cur_f_y;
    delete[] cur_c_y;
    delete[] cur_o_y;
    std::cout << "[EIGEN] Freeing rec_i_y\n";
    delete[] rec_i_y;
    delete[] rec_f_y;
    delete[] rec_c_y;
    delete[] rec_o_y;
#endif
  }
  delete[] i_cur_bias;
  delete[] f_cur_bias;
  delete[] c_cur_bias;
  delete[] o_cur_bias;
  delete[] i_sum;
  delete[] f_sum;
  delete[] c_sum;
  delete[] o_sum;
  delete[] i_gate;
  delete[] f_gate;
  delete[] o_gate;
  delete[] c_sum_tanh;
  delete[] c_tanh;
  delete[] c_lhs;
  delete[] c_rhs;
  delete[] c;
  delete[] rec_i_v_T;
  delete[] rec_f_v_T;
  delete[] rec_c_v_T;
  delete[] rec_o_v_T;
  delete[] cur_i_u_T;
  delete[] cur_f_u_T;
  delete[] cur_c_u_T;
  delete[] cur_o_u_T;
  delete[] rec_i_u_T;
  delete[] rec_f_u_T;
  delete[] rec_c_u_T;
  delete[] rec_o_u_T;
  delete[] cur_i_us;
  delete[] cur_f_us;
  delete[] cur_c_us;
  delete[] cur_o_us;
  delete[] rec_i_us;
  delete[] rec_f_us;
  delete[] rec_c_us;
  delete[] rec_o_us;
}

#ifdef __cplusplus
extern "C"
#endif
void SvdModelEigenUnbatched(const int verbose,
                            const bool use_blas,
                            float *x,
                            const int num_samples,
                            const int num_timesteps,
                            const int n_steps,
                            const int input_size,
                            const int hidden_size,
                            const float *cur_i_u,
                            const float *cur_i_s,
                            const float *cur_i_v,
                            const float *cur_f_u,
                            const float *cur_f_s,
                            const float *cur_f_v,
                            const float *cur_c_u,
                            const float *cur_c_s,
                            const float *cur_c_v,
                            const float *cur_o_u,
                            const float *cur_o_s,
                            const float *cur_o_v,
                            const float *rec_i_u,
                            const float *rec_i_s,
                            const float *rec_i_v,
                            const float *rec_f_u,
                            const float *rec_f_s,
                            const float *rec_f_v,
                            const float *rec_c_u,
                            const float *rec_c_s,
                            const float *rec_c_v,
                            const float *rec_o_u,
                            const float *rec_o_s,
                            const float *rec_o_v,
                            const float *bias_i,
                            const float *bias_f,
                            const float *bias_c,
                            const float *bias_o,
                            float *out) {
  // ===========================================================================
  // This C++ function is implementing the following python snippet:
  // (Note the extensive use of matrix multiplications)
  // 
  //   m = 6 # input size
  //   n = 8 # output size, i.e. LSTM hidden size (in Keras)
  //   b = 4 # batch size (or timesteps)
  //   k = 2 # num iterations, i.e. n_steps
  //   x = np.random.randn(m, b)
  //   u = np.random.randn(m, k)
  //   v = np.random.randn(n, k)
  //   s = np.random.randn(k)
  //   s2 = np.random.randn(k)
  //   
  //   s_mat = np.repeat(s, m).reshape(k, m).T
  //   us = u * s_mat
  //   
  //   y_not_batched = np.zeros((b, n))
  //   for i in range(b):
  //       ux = x.T[i] @ us
  //       y_not_batched[i] = ux @ v.T
  // ===========================================================================
  // ===========================================================================
  // Define: x.T @ US Matrixes
  // ===========================================================================
  // Current x
  float *cur_i_ux = new float[n_steps];
  float *cur_f_ux = new float[n_steps];
  float *cur_c_ux = new float[n_steps];
  float *cur_o_ux = new float[n_steps];
  // Recurrent x
  float *rec_i_uh = new float[n_steps];
  float *rec_f_uh = new float[n_steps];
  float *rec_c_uh = new float[n_steps];
  float *rec_o_uh = new float[n_steps];

  // Current y
  float *cur_i_y = new float[hidden_size];
  float *cur_f_y = new float[hidden_size];
  float *cur_c_y = new float[hidden_size];
  float *cur_o_y = new float[hidden_size];
  // Recurrent y
  float *rec_i_y = new float[hidden_size];
  float *rec_f_y = new float[hidden_size];
  float *rec_c_y = new float[hidden_size];
  float *rec_o_y = new float[hidden_size];

  // Output y
  float *i_cur_bias = new float[hidden_size];
  float *f_cur_bias = new float[hidden_size];
  float *c_cur_bias = new float[hidden_size];
  float *o_cur_bias = new float[hidden_size];

  float *i_sum = new float[hidden_size];
  float *f_sum = new float[hidden_size];
  float *c_sum = new float[hidden_size];
  float *o_sum = new float[hidden_size];

  float *i_gate = new float[hidden_size];
  float *f_gate = new float[hidden_size];
  float *o_gate = new float[hidden_size];
  float *c_sum_tanh = new float[hidden_size];
  float *c_tanh = new float[hidden_size];
  float *c_lhs = new float[hidden_size];
  float *c_rhs = new float[hidden_size];
  float *c = new float[hidden_size];

  // Current v Transposed
  float *cur_i_v_T = new float[hidden_size * n_steps];
  float *cur_f_v_T = new float[hidden_size * n_steps];
  float *cur_c_v_T = new float[hidden_size * n_steps];
  float *cur_o_v_T = new float[hidden_size * n_steps];
  // Recurrent v Transposed
  float *rec_i_v_T = new float[hidden_size * n_steps];
  float *rec_f_v_T = new float[hidden_size * n_steps];
  float *rec_c_v_T = new float[hidden_size * n_steps];
  float *rec_o_v_T = new float[hidden_size * n_steps];
  // ===========================================================================
  // Compute:
  // - S * U Matrix
  // ===========================================================================
  // Current us1
  float *cur_i_u_T = new float[n_steps * input_size];
  float *cur_f_u_T = new float[n_steps * input_size];
  float *cur_c_u_T = new float[n_steps * input_size];
  float *cur_o_u_T = new float[n_steps * input_size];
  // Recurrent us
  float *rec_i_u_T = new float[n_steps * hidden_size];
  float *rec_f_u_T = new float[n_steps * hidden_size];
  float *rec_c_u_T = new float[n_steps * hidden_size];
  float *rec_o_u_T = new float[n_steps * hidden_size];
  // Current us1
  float *cur_i_us = new float[n_steps * input_size];
  float *cur_f_us = new float[n_steps * input_size];
  float *cur_c_us = new float[n_steps * input_size];
  float *cur_o_us = new float[n_steps * input_size];
  // Recurrent us
  float *rec_i_us = new float[n_steps * hidden_size];
  float *rec_f_us = new float[n_steps * hidden_size];
  float *rec_c_us = new float[n_steps * hidden_size];
  float *rec_o_us = new float[n_steps * hidden_size];

  // ===========================================================================
  // NOTE: We need to 'transpose' u in order to generate the us matrix. This is
  // because the multiplication svd_mul() operates row-wise.
  // ===========================================================================
  // BEFORE TRANSPOSE: s.shape = (n_steps)
  // BEFORE TRANSPOSE: u.shape = (n_steps, input_size)
  // BEFORE TRANSPOSE: us.shape = (n_steps, input_size)
  svd_transpose(n_steps, input_size, cur_i_u, cur_i_u_T);
  svd_transpose(n_steps, input_size, cur_f_u, cur_f_u_T);
  svd_transpose(n_steps, input_size, cur_c_u, cur_c_u_T);
  svd_transpose(n_steps, input_size, cur_o_u, cur_o_u_T);
  svd_transpose(n_steps, hidden_size, rec_i_u, rec_i_u_T);
  svd_transpose(n_steps, hidden_size, rec_f_u, rec_f_u_T);
  svd_transpose(n_steps, hidden_size, rec_c_u, rec_c_u_T);
  svd_transpose(n_steps, hidden_size, rec_o_u, rec_o_u_T);
  for (int i = 0; i < input_size; ++i) {
    svd_mul(n_steps, &cur_i_u_T[i * n_steps], cur_i_s, &cur_i_us[i * n_steps]);
    svd_mul(n_steps, &cur_f_u_T[i * n_steps], cur_f_s, &cur_f_us[i * n_steps]);
    svd_mul(n_steps, &cur_c_u_T[i * n_steps], cur_c_s, &cur_c_us[i * n_steps]);
    svd_mul(n_steps, &cur_o_u_T[i * n_steps], cur_o_s, &cur_o_us[i * n_steps]);
  }
  for (int i = 0; i < hidden_size; ++i) {
    svd_mul(n_steps, &rec_i_u_T[i * n_steps], rec_i_s, &rec_i_us[i * n_steps]);
    svd_mul(n_steps, &rec_f_u_T[i * n_steps], rec_f_s, &rec_f_us[i * n_steps]);
    svd_mul(n_steps, &rec_c_u_T[i * n_steps], rec_c_s, &rec_c_us[i * n_steps]);
    svd_mul(n_steps, &rec_o_u_T[i * n_steps], rec_o_s, &rec_o_us[i * n_steps]);
  }
  // ===========================================================================
  // Transpose back current v and current u vectors.
  // ===========================================================================
  // From (input_size, n_steps) to (n_steps, input_size)
  svd_transpose(input_size, n_steps, cur_i_us, cur_i_u_T);
  svd_transpose(input_size, n_steps, cur_f_us, cur_f_u_T);
  svd_transpose(input_size, n_steps, cur_c_us, cur_c_u_T);
  svd_transpose(input_size, n_steps, cur_o_us, cur_o_u_T);
  // From (n_steps, hidden_size) to (hidden_size, n_steps)
  svd_transpose(n_steps, hidden_size, cur_i_v, cur_i_v_T);
  svd_transpose(n_steps, hidden_size, cur_f_v, cur_f_v_T);
  svd_transpose(n_steps, hidden_size, cur_c_v, cur_c_v_T);
  svd_transpose(n_steps, hidden_size, cur_o_v, cur_o_v_T); 
  // ===========================================================================
  // Transpose back recurrent v and recurrent u vectors.
  // ===========================================================================
  // From (hidden_size, n_steps) to (n_steps, hidden_size)
  svd_transpose(hidden_size, n_steps, rec_i_us, rec_i_u_T);
  svd_transpose(hidden_size, n_steps, rec_f_us, rec_f_u_T);
  svd_transpose(hidden_size, n_steps, rec_c_us, rec_c_u_T);
  svd_transpose(hidden_size, n_steps, rec_o_us, rec_o_u_T);
  // From (n_steps, hidden_size) to (hidden_size, n_steps)
  svd_transpose(n_steps, hidden_size, rec_i_v, rec_i_v_T);
  svd_transpose(n_steps, hidden_size, rec_f_v, rec_f_v_T);
  svd_transpose(n_steps, hidden_size, rec_c_v, rec_c_v_T);
  svd_transpose(n_steps, hidden_size, rec_o_v, rec_o_v_T);

  const int kSampleSize = num_timesteps * input_size;

#ifdef SDS_DESIGN
  perf_counter gemm_ctr;
  perf_counter gemv_ctr;
  perf_counter sw_ctr;
#else
#ifndef __SYNTHESIS__
  std::chrono::duration<double> total_time(0);
#endif
#endif
// =============================================================================
// @todo (11/02/2019): openmp NOT WORKING: threads still share resources,
// creating data races and producing incorrect results.
// =============================================================================
// #pragma omp parallel for schedule(static, 8)
// =============================================================================
#ifdef USE_BLAS
  const bool kAvailableBLAS = true;
#else
  const bool kAvailableBLAS = false;
#endif

#ifdef SDS_DESIGN
  sw_ctr.start();
#elif !defined(__SYNTHESIS__)
  auto begin_timestep = std::chrono::high_resolution_clock::now();
#endif
  if (use_blas && kAvailableBLAS) {
#ifdef USE_BLAS
    for (int i = 0; i < num_samples; ++i) {
      svd_set(hidden_size, (float)0., c);
      svd_set(hidden_size, (float)0., &out[i * hidden_size]);

      for (int j = 0; j < num_timesteps; ++j) {
        // =======================================================================
        // Current LSTM gates
        // =======================================================================
        // Current us.T @ x: (input_size, n_steps).T @ (input_size, 1)
        svd_cpu_gemv(CblasNoTrans, n_steps, input_size, (float)1., cur_i_u_T, &x[i * kSampleSize + j * input_size], (float)0., cur_i_ux);
        svd_cpu_gemv(CblasNoTrans, n_steps, input_size, (float)1., cur_f_u_T, &x[i * kSampleSize + j * input_size], (float)0., cur_f_ux);
        svd_cpu_gemv(CblasNoTrans, n_steps, input_size, (float)1., cur_c_u_T, &x[i * kSampleSize + j * input_size], (float)0., cur_c_ux);
        svd_cpu_gemv(CblasNoTrans, n_steps, input_size, (float)1., cur_o_u_T, &x[i * kSampleSize + j * input_size], (float)0., cur_o_ux);

        // NOTE: We can skip later bias addition if we memcpy the bias into cur_y,
        // that's because gemv performs: C = alpha * A * B + beta * C, notice
        // that beta is indeed set to 1 here.
        svd_copy(hidden_size, bias_i, i_cur_bias);
        svd_copy(hidden_size, bias_f, f_cur_bias);
        svd_copy(hidden_size, bias_c, c_cur_bias);
        svd_copy(hidden_size, bias_o, o_cur_bias);
        // Current v.T @ ux: (n_steps, hidden_size).T @ (n_steps, 1)
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., cur_i_v_T, cur_i_ux, (float)1., i_cur_bias);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., cur_f_v_T, cur_f_ux, (float)1., f_cur_bias);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., cur_c_v_T, cur_c_ux, (float)1., c_cur_bias);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., cur_o_v_T, cur_o_ux, (float)1., o_cur_bias);

        // =======================================================================
        // Recurrent LSTM gates
        // =======================================================================
        // Recurrent us.T @ h: (hidden_size, n_steps).T @ (hidden_size, 1)
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_i_u_T, &out[i * hidden_size], (float)0., rec_i_uh);
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_f_u_T, &out[i * hidden_size], (float)0., rec_f_uh);
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_c_u_T, &out[i * hidden_size], (float)0., rec_c_uh);
        svd_cpu_gemv(CblasNoTrans, n_steps, hidden_size, (float)1., rec_o_u_T, &out[i * hidden_size], (float)0., rec_o_uh);

        // Recurrent v.T @ uh: (n_steps, hidden_size).T @ (n_steps, 1)
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_i_v_T, rec_i_uh, (float)0., rec_i_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_f_v_T, rec_f_uh, (float)0., rec_f_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_c_v_T, rec_c_uh, (float)0., rec_c_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, n_steps, (float)1., rec_o_v_T, rec_o_uh, (float)0., rec_o_y);
        // =======================================================================
        // Non linearities
        // =======================================================================
        // NOTE: We can skip the bias addition if we previously memcpy-ed
        // the bias into cur_bias, that's because gemv performs: C = alpha * A * B + beta * C
        // and C can be initialized to the bias.
        svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
        svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
        svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
        svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

        svd_hard_sigmoid(hidden_size, i_sum, i_gate);
        svd_hard_sigmoid(hidden_size, f_sum, f_gate);
        svd_hard_sigmoid(hidden_size, o_sum, o_gate);
        svd_tanh(hidden_size, c_sum, c_sum_tanh);
        svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
        svd_mul(hidden_size, c, f_gate, c_rhs);

        svd_add(hidden_size, c_lhs, c_rhs, c);
        svd_tanh(hidden_size, c, c_tanh);
        svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
      }
    }
#endif // end USE_BLAS
  } else {
// #pragma omp parallel for num_threads(8) schedule(dynamic)
    for (int i = 0; i < num_samples; ++i) {
      svd_set(hidden_size, (float)0., c);
      svd_set(hidden_size, (float)0., &out[i * hidden_size]);

      for (int j = 0; j < num_timesteps; ++j) {
        // =======================================================================
        // Current LSTM gates
        // =======================================================================
        // NOTE: in this unbatched version, current and recurrent gates execution
        // is simmetrical, i.e. same transposed matrices logic.
        // =======================================================================
        // us.T @ x
        hls_gemv<float, float>(n_steps, input_size, cur_i_u_T, &x[i * kSampleSize + j * input_size], cur_i_ux);
        hls_gemv<float, float>(n_steps, input_size, cur_f_u_T, &x[i * kSampleSize + j * input_size], cur_f_ux);
        hls_gemv<float, float>(n_steps, input_size, cur_c_u_T, &x[i * kSampleSize + j * input_size], cur_c_ux);
        hls_gemv<float, float>(n_steps, input_size, cur_o_u_T, &x[i * kSampleSize + j * input_size], cur_o_ux);
   
        // v.T @ xus
        hls_gemv<float, float>(hidden_size, n_steps, cur_i_v_T, cur_i_ux, cur_i_y);
        hls_gemv<float, float>(hidden_size, n_steps, cur_f_v_T, cur_f_ux, cur_f_y);
        hls_gemv<float, float>(hidden_size, n_steps, cur_c_v_T, cur_c_ux, cur_c_y);
        hls_gemv<float, float>(hidden_size, n_steps, cur_o_v_T, cur_o_ux, cur_o_y);

        // =======================================================================
        // Recurrent LSTM gates
        // =======================================================================
        // us.T @ h
        hls_gemv<float, float>(n_steps, hidden_size, rec_i_u_T, &out[i * hidden_size], rec_i_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_f_u_T, &out[i * hidden_size], rec_f_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_c_u_T, &out[i * hidden_size], rec_c_uh);
        hls_gemv<float, float>(n_steps, hidden_size, rec_o_u_T, &out[i * hidden_size], rec_o_uh);
        // v.T @ hus
        hls_gemv<float, float>(hidden_size, n_steps, rec_i_v_T, rec_i_uh, rec_i_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_f_v_T, rec_f_uh, rec_f_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_c_v_T, rec_c_uh, rec_c_y);
        hls_gemv<float, float>(hidden_size, n_steps, rec_o_v_T, rec_o_uh, rec_o_y);
        // =======================================================================
        // Non linearities
        // =======================================================================
        // NOTE: We can skip the following bias addition if we previously memcpy
        // the bias into cur_bias, that's because gemv performs: C = alpha * A * B + beta * C
        // and C can be initialized to the bias.
        svd_add(hidden_size, cur_i_y, bias_i, i_cur_bias);
        svd_add(hidden_size, cur_f_y, bias_f, f_cur_bias);
        svd_add(hidden_size, cur_c_y, bias_c, c_cur_bias);
        svd_add(hidden_size, cur_o_y, bias_o, o_cur_bias);

        svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
        svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
        svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
        svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

        svd_hard_sigmoid(hidden_size, i_sum, i_gate);
        svd_hard_sigmoid(hidden_size, f_sum, f_gate);
        svd_hard_sigmoid(hidden_size, o_sum, o_gate);
        svd_tanh(hidden_size, c_sum, c_sum_tanh);
        svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
        svd_mul(hidden_size, c, f_gate, c_rhs);

        svd_add(hidden_size, c_lhs, c_rhs, c);
        svd_tanh(hidden_size, c, c_tanh);
        svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
      }
    }
  }
#ifdef SDS_DESIGN
  sw_ctr.stop();
#elif !defined(__SYNTHESIS__)
  auto end_timestep = std::chrono::high_resolution_clock::now();
  total_time += end_timestep - begin_timestep;
#endif
  // ===========================================================================
  // NOTE: We are NOT taking into account the time it takes to both setup the u @ s
  // matrices and perform their transpositions because these operations can be
  // done "offline", i.e. can be stored in that form already, performance-wise.
  // ===========================================================================
  if (verbose == 1) {
#ifdef SDS_DESIGN
    auto sw_cycles = sw_ctr.avg_cpu_cycles();
    auto sw_freq = sds_clock_frequency();
    std::cout << "Frequency: " << sw_freq << " ticks/second\n";
#ifdef USE_BLAS
    if (use_blas) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
#endif
    std::cout << "Unbatched SVD: Total CPU cycles: " << std::setprecision(12) << sw_cycles << "\n";
#elif !defined(__SYNTHESIS__)
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(total_time).count();
    auto duration_us = duration_ns / 1000.0;
    auto duration_ms = duration_us / 1000.0;
    auto duration_s = duration_ms / 1000.0;
    #ifdef USE_BLAS
    if (use_blas) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
#endif
    std::cout << "Unbatched SVD: Total time: " << duration_ms
              << " ms (" << duration_s << " s)."<< "\n";
#ifdef USE_BLAS
    if (use_blas) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
#endif
    std::cout << "Unbatched SVD: Average time per sample: " << duration_ms / num_samples
              << " ms (" << duration_s / num_samples << " s)."<< "\n";
#ifdef USE_BLAS
    if (use_blas) {
      std::cout << "[BLAS version] ";
    } else {
      std::cout << "[no BLAS] ";
    }
#endif
    std::cout << "Unbatched SVD: Average time per timesteps: " << duration_ms / num_samples / num_timesteps
              << " ms (" << duration_s / num_samples / num_timesteps << " s)."<< "\n";
#endif
  }
  delete[] cur_i_ux;
  delete[] cur_f_ux;
  delete[] cur_c_ux;
  delete[] cur_o_ux;
  delete[] rec_i_uh;
  delete[] rec_f_uh;
  delete[] rec_c_uh;
  delete[] rec_o_uh;
  delete[] cur_i_y;
  delete[] cur_f_y;
  delete[] cur_c_y;
  delete[] cur_o_y;
  delete[] rec_i_y;
  delete[] rec_f_y;
  delete[] rec_c_y;
  delete[] rec_o_y;
  delete[] i_cur_bias;
  delete[] f_cur_bias;
  delete[] c_cur_bias;
  delete[] o_cur_bias;
  delete[] i_sum;
  delete[] f_sum;
  delete[] c_sum;
  delete[] o_sum;
  delete[] i_gate;
  delete[] f_gate;
  delete[] o_gate;
  delete[] c_sum_tanh;
  delete[] c_tanh;
  delete[] c_lhs;
  delete[] c_rhs;
  delete[] c;
  delete[] rec_i_v_T;
  delete[] rec_f_v_T;
  delete[] rec_c_v_T;
  delete[] rec_o_v_T;
  delete[] cur_i_u_T;
  delete[] cur_f_u_T;
  delete[] cur_c_u_T;
  delete[] cur_o_u_T;
  delete[] rec_i_u_T;
  delete[] rec_f_u_T;
  delete[] rec_c_u_T;
  delete[] rec_o_u_T;
  delete[] cur_i_us;
  delete[] cur_f_us;
  delete[] cur_c_us;
  delete[] cur_o_us;
  delete[] rec_i_us;
  delete[] rec_f_us;
  delete[] rec_c_us;
  delete[] rec_o_us;
}


#if 0
#ifdef __cplusplus
extern "C"
#endif
void SvdModel2LstmHardware(const int verbose,
                           const bool use_blas,
                           const int type, // 0:float, 1:fix8, 2:fix16
                           const float *x1,
                           const float *x2,
                           const int num_samples,
                           const int num_timesteps,
                           const int n_steps,
                           const int input_size,
                           const int hidden_size,
                           const float *cur_i_u,
                           const float *cur_i_s1,
                           const float *cur_i_s2,
                           const float *cur_i_v,
                           const float *cur_f_u,
                           const float *cur_f_s1,
                           const float *cur_f_s2,
                           const float *cur_f_v,
                           const float *cur_c_u,
                           const float *cur_c_s1,
                           const float *cur_c_s2,
                           const float *cur_c_v,
                           const float *cur_o_u,
                           const float *cur_o_s1,
                           const float *cur_o_s2,
                           const float *cur_o_v,
                           const float *rec_i_u,
                           const float *rec_i_s1,
                           const float *rec_i_s2,
                           const float *rec_i_v,
                           const float *rec_f_u,
                           const float *rec_f_s1,
                           const float *rec_f_s2,
                           const float *rec_f_v,
                           const float *rec_c_u,
                           const float *rec_c_s1,
                           const float *rec_c_s2,
                           const float *rec_c_v,
                           const float *rec_o_u,
                           const float *rec_o_s1,
                           const float *rec_o_s2,
                           const float *rec_o_v,
                           const float *bias_i,
                           const float *bias_f,
                           const float *bias_c,
                           const float *bias_o,
                           float *out1,
                           float *out2) {
  switch(type) {
    case 0: {
        // =====================================================================
        // Float32
        // =====================================================================
        
          int InputSize,
          int HiddenSize,
          int NumIter,
          int Tu,
          int ZTu,
          int Tv,
          int ZTv,
          int NumTimesteps
        SvdModel2LstmTemplated<float, float>(x1,
                      x2,
                      cur_i_u,
                      cur_i_s,
                      cur_i_v,
                      const ap_uint<Tu> cur_i_uz[NumIter],
                      const ap_uint<Tv> cur_i_vz[NumIter],
                      const DataW cur_f_u[InputSize / Tu * (Tu - ZTu)],
                      const DataW cur_f_s[2 * NumIter],
                      const DataW cur_f_v[HiddenSize / Tv * (Tv - ZTv)],
                      const ap_uint<Tu> cur_f_uz[NumIter],
                      const ap_uint<Tv> cur_f_vz[NumIter],
                      const DataW cur_c_u[InputSize / Tu * (Tu - ZTu)],
                      const DataW cur_c_s[2 * NumIter],
                      const DataW cur_c_v[HiddenSize / Tv * (Tv - ZTv)],
                      const ap_uint<Tu> cur_c_uz[NumIter],
                      const ap_uint<Tv> cur_c_vz[NumIter],
                      const DataW cur_o_u[InputSize / Tu * (Tu - ZTu)],
                      const DataW cur_o_s[2 * NumIter],
                      const DataW cur_o_v[HiddenSize / Tv * (Tv - ZTv)],
                      const ap_uint<Tu> cur_o_uz[NumIter],
                      const ap_uint<Tv> cur_o_vz[NumIter],
                      const DataW rec_i_u[HiddenSize / Tu * (Tu - ZTu)],
                      const DataW rec_i_s[2 * NumIter],
                      const DataW rec_i_v[HiddenSize / Tv * (Tv - ZTv)],
                      const ap_uint<Tu> rec_i_uz[NumIter],
                      const ap_uint<Tv> rec_i_vz[NumIter],
                      const DataW rec_f_u[HiddenSize / Tu * (Tu - ZTu)],
                      const DataW rec_f_s[2 * NumIter],
                      const DataW rec_f_v[HiddenSize / Tv * (Tv - ZTv)],
                      const ap_uint<Tu> rec_f_uz[NumIter],
                      const ap_uint<Tv> rec_f_vz[NumIter],
                      const DataW rec_c_u[HiddenSize / Tu * (Tu - ZTu)],
                      const DataW rec_c_s[2 * NumIter],
                      const DataW rec_c_v[HiddenSize / Tv * (Tv - ZTv)],
                      const ap_uint<Tu> rec_c_uz[NumIter],
                      const ap_uint<Tv> rec_c_vz[NumIter],
                      const DataW rec_o_u[HiddenSize / Tu * (Tu - ZTu)],
                      const DataW rec_o_s[2 * NumIter],
                      const DataW rec_o_v[HiddenSize / Tv * (Tv - ZTv)],
                      const ap_uint<Tu> rec_o_uz[NumIter],
                      const ap_uint<Tv> rec_o_vz[NumIter],
                      const DataW bias1[4 * HiddenSize],
                      const DataW bias2[4 * HiddenSize],
                      // const DataW bias1_i[HiddenSize],
                      // const DataW bias1_f[HiddenSize],
                      // const DataW bias1_c[HiddenSize],
                      // const DataW bias1_o[HiddenSize],
                      // const DataW bias2_i[HiddenSize],
                      // const DataW bias2_f[HiddenSize],
                      // const DataW bias2_c[HiddenSize],
                      // const DataW bias2_o[HiddenSize],
                      const DataA c1_prev[HiddenSize],
                      const DataA c2_prev[HiddenSize],
                      const DataA h1_prev[HiddenSize],
                      const DataA h2_prev[HiddenSize],
                      DataA c1_curr[HiddenSize],
                      DataA c2_curr[HiddenSize],
                      DataA h1_curr[HiddenSize],
                      DataA h2_curr[HiddenSize]);
        SvdModelSoftwareUnbatched(verbose, use_blas, x, num_samples, num_timesteps,
                              n_steps, input_size, hidden_size, cur_i_u, cur_i_s,
                              cur_i_v, cur_f_u, cur_f_s, cur_f_v, cur_c_u,
                              cur_c_s, cur_c_v, cur_o_u, cur_o_s, cur_o_v,
                              rec_i_u, rec_i_s, rec_i_v, rec_f_u, rec_f_s,
                              rec_f_v, rec_c_u, rec_c_s, rec_c_v, rec_o_u,
                              rec_o_s, rec_o_v, bias_i, bias_f, bias_c,
                              bias_o, out);
        }
      break;
    case 1: {
        // =====================================================================
        // Fixed point arrays (8 bit)
        // =====================================================================
        Fix8D *x_fix = new Fix8D[num_samples * num_timesteps * input_size];
        Fix8D *cur_i_u_fix = new Fix8D[n_steps * input_size];
        Fix8D *cur_i_s_fix = new Fix8D[n_steps];
        Fix8D *cur_i_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *cur_f_u_fix = new Fix8D[n_steps * input_size];
        Fix8D *cur_f_s_fix = new Fix8D[n_steps];
        Fix8D *cur_f_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *cur_c_u_fix = new Fix8D[n_steps * input_size];
        Fix8D *cur_c_s_fix = new Fix8D[n_steps];
        Fix8D *cur_c_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *cur_o_u_fix = new Fix8D[n_steps * input_size];
        Fix8D *cur_o_s_fix = new Fix8D[n_steps];
        Fix8D *cur_o_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_i_u_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_i_s_fix = new Fix8D[n_steps];
        Fix8D *rec_i_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_f_u_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_f_s_fix = new Fix8D[n_steps];
        Fix8D *rec_f_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_c_u_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_c_s_fix = new Fix8D[n_steps];
        Fix8D *rec_c_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_o_u_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *rec_o_s_fix = new Fix8D[n_steps];
        Fix8D *rec_o_v_fix = new Fix8D[n_steps * hidden_size];
        Fix8D *bias_i_fix = new Fix8D[hidden_size];
        Fix8D *bias_f_fix = new Fix8D[hidden_size];
        Fix8D *bias_c_fix = new Fix8D[hidden_size];
        Fix8D *bias_o_fix = new Fix8D[hidden_size];
        Fix8D *out_fix = new Fix8D[num_samples * hidden_size];
        // =====================================================================
        // Copy and cast to fixed point
        // =====================================================================
        hls_copy_cast(num_samples * num_timesteps * input_size, x, x_fix);
        hls_copy_cast(n_steps * input_size, cur_i_u, cur_i_u_fix);
        hls_copy_cast(n_steps, cur_i_s, cur_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_i_v, cur_i_v_fix);
        hls_copy_cast(n_steps * input_size, cur_f_u, cur_f_u_fix);
        hls_copy_cast(n_steps, cur_f_s, cur_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_f_v, cur_f_v_fix);
        hls_copy_cast(n_steps * input_size, cur_c_u, cur_c_u_fix);
        hls_copy_cast(n_steps, cur_c_s, cur_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_c_v, cur_c_v_fix);
        hls_copy_cast(n_steps * input_size, cur_o_u, cur_o_u_fix);
        hls_copy_cast(n_steps, cur_o_s, cur_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_o_v, cur_o_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_u, rec_i_u_fix);
        hls_copy_cast(n_steps, rec_i_s, rec_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_v, rec_i_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_u, rec_f_u_fix);
        hls_copy_cast(n_steps, rec_f_s, rec_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_v, rec_f_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_u, rec_c_u_fix);
        hls_copy_cast(n_steps, rec_c_s, rec_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_v, rec_c_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_u, rec_o_u_fix);
        hls_copy_cast(n_steps, rec_o_s, rec_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_v, rec_o_v_fix);
        hls_copy_cast(hidden_size, bias_i, bias_i_fix);
        hls_copy_cast(hidden_size, bias_f, bias_f_fix);
        hls_copy_cast(hidden_size, bias_c, bias_c_fix);
        hls_copy_cast(hidden_size, bias_o, bias_o_fix);
        // =====================================================================
        // Call function
        // =====================================================================
        SvdModel2LstmFix8(verbose, x_fix, num_samples, num_timesteps, n_steps, input_size,
                          hidden_size, cur_i_u_fix, cur_i_s_fix, cur_i_v_fix,
                          cur_f_u_fix, cur_f_s_fix, cur_f_v_fix, cur_c_u_fix,
                          cur_c_s_fix, cur_c_v_fix, cur_o_u_fix, cur_o_s_fix,
                          cur_o_v_fix, rec_i_u_fix, rec_i_s_fix, rec_i_v_fix,
                          rec_f_u_fix, rec_f_s_fix, rec_f_v_fix, rec_c_u_fix,
                          rec_c_s_fix, rec_c_v_fix, rec_o_u_fix, rec_o_s_fix,
                          rec_o_v_fix, bias_i_fix, bias_f_fix, bias_c_fix,
                          bias_o_fix, out_fix);
        // =====================================================================
        // Writeback
        // =====================================================================
        hls_copy_cast(num_samples * hidden_size, out_fix, out);
        // =====================================================================
        // Cleanup
        // =====================================================================
        delete[] x_fix;
        delete[] cur_i_u_fix;
        delete[] cur_i_s_fix;
        delete[] cur_i_v_fix;
        delete[] cur_f_u_fix;
        delete[] cur_f_s_fix;
        delete[] cur_f_v_fix;
        delete[] cur_c_u_fix;
        delete[] cur_c_s_fix;
        delete[] cur_c_v_fix;
        delete[] cur_o_u_fix;
        delete[] cur_o_s_fix;
        delete[] cur_o_v_fix;
        delete[] rec_i_u_fix;
        delete[] rec_i_s_fix;
        delete[] rec_i_v_fix;
        delete[] rec_f_u_fix;
        delete[] rec_f_s_fix;
        delete[] rec_f_v_fix;
        delete[] rec_c_u_fix;
        delete[] rec_c_s_fix;
        delete[] rec_c_v_fix;
        delete[] rec_o_u_fix;
        delete[] rec_o_s_fix;
        delete[] rec_o_v_fix;
        delete[] bias_i_fix;
        delete[] bias_f_fix;
        delete[] bias_c_fix;
        delete[] bias_o_fix;
        delete[] out_fix;
      }
      break;
    case 2: {
        // =====================================================================
        // Fixed point arrays (16 bit)
        // =====================================================================
        Fix16D *x_fix = new Fix16D[num_samples * num_timesteps * input_size];
        Fix16D *cur_i_u_fix = new Fix16D[n_steps * input_size];
        Fix16D *cur_i_s_fix = new Fix16D[n_steps];
        Fix16D *cur_i_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *cur_f_u_fix = new Fix16D[n_steps * input_size];
        Fix16D *cur_f_s_fix = new Fix16D[n_steps];
        Fix16D *cur_f_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *cur_c_u_fix = new Fix16D[n_steps * input_size];
        Fix16D *cur_c_s_fix = new Fix16D[n_steps];
        Fix16D *cur_c_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *cur_o_u_fix = new Fix16D[n_steps * input_size];
        Fix16D *cur_o_s_fix = new Fix16D[n_steps];
        Fix16D *cur_o_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_i_u_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_i_s_fix = new Fix16D[n_steps];
        Fix16D *rec_i_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_f_u_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_f_s_fix = new Fix16D[n_steps];
        Fix16D *rec_f_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_c_u_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_c_s_fix = new Fix16D[n_steps];
        Fix16D *rec_c_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_o_u_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *rec_o_s_fix = new Fix16D[n_steps];
        Fix16D *rec_o_v_fix = new Fix16D[n_steps * hidden_size];
        Fix16D *bias_i_fix = new Fix16D[hidden_size];
        Fix16D *bias_f_fix = new Fix16D[hidden_size];
        Fix16D *bias_c_fix = new Fix16D[hidden_size];
        Fix16D *bias_o_fix = new Fix16D[hidden_size];
        Fix16D *out_fix = new Fix16D[num_samples * hidden_size];
        // =====================================================================
        // Copy and cast to fixed point
        // =====================================================================
        hls_copy_cast(num_samples * num_timesteps * input_size, x, x_fix);
        hls_copy_cast(n_steps * input_size, cur_i_u, cur_i_u_fix);
        hls_copy_cast(n_steps, cur_i_s, cur_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_i_v, cur_i_v_fix);
        hls_copy_cast(n_steps * input_size, cur_f_u, cur_f_u_fix);
        hls_copy_cast(n_steps, cur_f_s, cur_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_f_v, cur_f_v_fix);
        hls_copy_cast(n_steps * input_size, cur_c_u, cur_c_u_fix);
        hls_copy_cast(n_steps, cur_c_s, cur_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_c_v, cur_c_v_fix);
        hls_copy_cast(n_steps * input_size, cur_o_u, cur_o_u_fix);
        hls_copy_cast(n_steps, cur_o_s, cur_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_o_v, cur_o_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_u, rec_i_u_fix);
        hls_copy_cast(n_steps, rec_i_s, rec_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_v, rec_i_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_u, rec_f_u_fix);
        hls_copy_cast(n_steps, rec_f_s, rec_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_v, rec_f_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_u, rec_c_u_fix);
        hls_copy_cast(n_steps, rec_c_s, rec_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_v, rec_c_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_u, rec_o_u_fix);
        hls_copy_cast(n_steps, rec_o_s, rec_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_v, rec_o_v_fix);
        hls_copy_cast(hidden_size, bias_i, bias_i_fix);
        hls_copy_cast(hidden_size, bias_f, bias_f_fix);
        hls_copy_cast(hidden_size, bias_c, bias_c_fix);
        hls_copy_cast(hidden_size, bias_o, bias_o_fix);
        // =====================================================================
        // Call function
        // =====================================================================
        SvdModel2LstmFix16(verbose, x_fix, num_samples, num_timesteps, n_steps,
                           input_size, hidden_size, cur_i_u_fix, cur_i_s_fix,
                           cur_i_v_fix, cur_f_u_fix, cur_f_s_fix, cur_f_v_fix,
                           cur_c_u_fix, cur_c_s_fix, cur_c_v_fix, cur_o_u_fix,
                           cur_o_s_fix, cur_o_v_fix, rec_i_u_fix, rec_i_s_fix,
                           rec_i_v_fix, rec_f_u_fix, rec_f_s_fix, rec_f_v_fix,
                           rec_c_u_fix, rec_c_s_fix, rec_c_v_fix, rec_o_u_fix,
                           rec_o_s_fix, rec_o_v_fix, bias_i_fix, bias_f_fix,
                           bias_c_fix, bias_o_fix, out_fix);
        // =====================================================================
        // Writeback
        // =====================================================================
        hls_copy_cast(num_samples * hidden_size, out_fix, out);
        // =====================================================================
        // Cleanup
        // =====================================================================
        delete[] x_fix;
        delete[] cur_i_u_fix;
        delete[] cur_i_s_fix;
        delete[] cur_i_v_fix;
        delete[] cur_f_u_fix;
        delete[] cur_f_s_fix;
        delete[] cur_f_v_fix;
        delete[] cur_c_u_fix;
        delete[] cur_c_s_fix;
        delete[] cur_c_v_fix;
        delete[] cur_o_u_fix;
        delete[] cur_o_s_fix;
        delete[] cur_o_v_fix;
        delete[] rec_i_u_fix;
        delete[] rec_i_s_fix;
        delete[] rec_i_v_fix;
        delete[] rec_f_u_fix;
        delete[] rec_f_s_fix;
        delete[] rec_f_v_fix;
        delete[] rec_c_u_fix;
        delete[] rec_c_s_fix;
        delete[] rec_c_v_fix;
        delete[] rec_o_u_fix;
        delete[] rec_o_s_fix;
        delete[] rec_o_v_fix;
        delete[] bias_i_fix;
        delete[] bias_f_fix;
        delete[] bias_c_fix;
        delete[] bias_o_fix;
        delete[] out_fix;
      }
      break;
    case 3: {
      // =====================================================================
        // Half floating point (16 bit)
        // =====================================================================
        HalfD *x_fix = new HalfD[num_samples * num_timesteps * input_size];
        HalfD *cur_i_u_fix = new HalfD[n_steps * input_size];
        HalfD *cur_i_s_fix = new HalfD[n_steps];
        HalfD *cur_i_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *cur_f_u_fix = new HalfD[n_steps * input_size];
        HalfD *cur_f_s_fix = new HalfD[n_steps];
        HalfD *cur_f_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *cur_c_u_fix = new HalfD[n_steps * input_size];
        HalfD *cur_c_s_fix = new HalfD[n_steps];
        HalfD *cur_c_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *cur_o_u_fix = new HalfD[n_steps * input_size];
        HalfD *cur_o_s_fix = new HalfD[n_steps];
        HalfD *cur_o_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_i_u_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_i_s_fix = new HalfD[n_steps];
        HalfD *rec_i_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_f_u_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_f_s_fix = new HalfD[n_steps];
        HalfD *rec_f_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_c_u_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_c_s_fix = new HalfD[n_steps];
        HalfD *rec_c_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_o_u_fix = new HalfD[n_steps * hidden_size];
        HalfD *rec_o_s_fix = new HalfD[n_steps];
        HalfD *rec_o_v_fix = new HalfD[n_steps * hidden_size];
        HalfD *bias_i_fix = new HalfD[hidden_size];
        HalfD *bias_f_fix = new HalfD[hidden_size];
        HalfD *bias_c_fix = new HalfD[hidden_size];
        HalfD *bias_o_fix = new HalfD[hidden_size];
        HalfD *out_fix = new HalfD[num_samples * hidden_size];
        // =====================================================================
        // Copy and cast to fixed point
        // =====================================================================
        // std::cout << "Starting converting float to half\n";
        hls_copy_cast(num_samples * num_timesteps * input_size, x, x_fix);
        hls_copy_cast(n_steps * input_size, cur_i_u, cur_i_u_fix);
        hls_copy_cast(n_steps, cur_i_s, cur_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_i_v, cur_i_v_fix);
        hls_copy_cast(n_steps * input_size, cur_f_u, cur_f_u_fix);
        hls_copy_cast(n_steps, cur_f_s, cur_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_f_v, cur_f_v_fix);
        hls_copy_cast(n_steps * input_size, cur_c_u, cur_c_u_fix);
        hls_copy_cast(n_steps, cur_c_s, cur_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_c_v, cur_c_v_fix);
        hls_copy_cast(n_steps * input_size, cur_o_u, cur_o_u_fix);
        hls_copy_cast(n_steps, cur_o_s, cur_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, cur_o_v, cur_o_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_u, rec_i_u_fix);
        hls_copy_cast(n_steps, rec_i_s, rec_i_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_i_v, rec_i_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_u, rec_f_u_fix);
        hls_copy_cast(n_steps, rec_f_s, rec_f_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_f_v, rec_f_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_u, rec_c_u_fix);
        hls_copy_cast(n_steps, rec_c_s, rec_c_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_c_v, rec_c_v_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_u, rec_o_u_fix);
        hls_copy_cast(n_steps, rec_o_s, rec_o_s_fix);
        hls_copy_cast(n_steps * hidden_size, rec_o_v, rec_o_v_fix);
        hls_copy_cast(hidden_size, bias_i, bias_i_fix);
        hls_copy_cast(hidden_size, bias_f, bias_f_fix);
        hls_copy_cast(hidden_size, bias_c, bias_c_fix);
        hls_copy_cast(hidden_size, bias_o, bias_o_fix);
        // =====================================================================
        // Call function
        // =====================================================================
        // std::cout << "Starting SvdModel2LstmHalf\n";
        SvdModel2LstmHalf(verbose, x_fix, num_samples, num_timesteps, n_steps,
                           input_size, hidden_size, cur_i_u_fix, cur_i_s_fix,
                           cur_i_v_fix, cur_f_u_fix, cur_f_s_fix, cur_f_v_fix,
                           cur_c_u_fix, cur_c_s_fix, cur_c_v_fix, cur_o_u_fix,
                           cur_o_s_fix, cur_o_v_fix, rec_i_u_fix, rec_i_s_fix,
                           rec_i_v_fix, rec_f_u_fix, rec_f_s_fix, rec_f_v_fix,
                           rec_c_u_fix, rec_c_s_fix, rec_c_v_fix, rec_o_u_fix,
                           rec_o_s_fix, rec_o_v_fix, bias_i_fix, bias_f_fix,
                           bias_c_fix, bias_o_fix, out_fix);
        // =====================================================================
        // Writeback
        // =====================================================================
        hls_copy_cast(num_samples * hidden_size, out_fix, out);
        // =====================================================================
        // Cleanup
        // =====================================================================
        delete[] x_fix;
        delete[] cur_i_u_fix;
        delete[] cur_i_s_fix;
        delete[] cur_i_v_fix;
        delete[] cur_f_u_fix;
        delete[] cur_f_s_fix;
        delete[] cur_f_v_fix;
        delete[] cur_c_u_fix;
        delete[] cur_c_s_fix;
        delete[] cur_c_v_fix;
        delete[] cur_o_u_fix;
        delete[] cur_o_s_fix;
        delete[] cur_o_v_fix;
        delete[] rec_i_u_fix;
        delete[] rec_i_s_fix;
        delete[] rec_i_v_fix;
        delete[] rec_f_u_fix;
        delete[] rec_f_s_fix;
        delete[] rec_f_v_fix;
        delete[] rec_c_u_fix;
        delete[] rec_c_s_fix;
        delete[] rec_c_v_fix;
        delete[] rec_o_u_fix;
        delete[] rec_o_s_fix;
        delete[] rec_o_v_fix;
        delete[] bias_i_fix;
        delete[] bias_f_fix;
        delete[] bias_c_fix;
        delete[] bias_o_fix;
        delete[] out_fix;
      }
      break;
    default: {
        // =====================================================================
        // Float32
        // =====================================================================
        SvdModelSoftwareUnbatched(verbose, use_blas, x, num_samples, num_timesteps,
                              n_steps, input_size, hidden_size, cur_i_u, cur_i_s,
                              cur_i_v, cur_f_u, cur_f_s, cur_f_v, cur_c_u,
                              cur_c_s, cur_c_v, cur_o_u, cur_o_s, cur_o_v,
                              rec_i_u, rec_i_s, rec_i_v, rec_f_u, rec_f_s,
                              rec_f_v, rec_c_u, rec_c_s, rec_c_v, rec_o_u,
                              rec_o_s, rec_o_v, bias_i, bias_f, bias_c,
                              bias_o, out);
      }
      break;
  }
}
#endif

template <typename T>
void print_vect(const int size, const int num_elem_to_print, T *v) {
  for (int i = 0; i < std::min(size, num_elem_to_print); ++i) {
    std::cout << v[i] << " ";
  }
  std::cout << "... " << v[size - 1] << "\n";
}

#if 1
#ifdef __cplusplus
extern "C"
#endif
void SvdModel2LstmSoftwareBatched(const int verbose,
                             const bool use_blas,
                             const float *x, // (num_samples, num_inputs, num_timesteps, input_size)
                             const int num_inputs,
                             const int num_samples,
                             const int num_timesteps,
                             const int n_steps,
                             const int input_size,
                             const int hidden_size,
                             const float *cur_i_u,
                             const float *cur_i_s,
                             const float *cur_i_v,
                             const float *cur_f_u,
                             const float *cur_f_s,
                             const float *cur_f_v,
                             const float *cur_c_u,
                             const float *cur_c_s,
                             const float *cur_c_v,
                             const float *cur_o_u,
                             const float *cur_o_s,
                             const float *cur_o_v,
                             const float *rec_i_u,
                             const float *rec_i_s,
                             const float *rec_i_v,
                             const float *rec_f_u,
                             const float *rec_f_s,
                             const float *rec_f_v,
                             const float *rec_c_u,
                             const float *rec_c_s,
                             const float *rec_c_v,
                             const float *rec_o_u,
                             const float *rec_o_s,
                             const float *rec_o_v,
                             const float *bias_i,
                             const float *bias_f,
                             const float *bias_c,
                             const float *bias_o,
                             float *out) {
  // ===========================================================================
  // This C++ function is implementing the following python snippet:
  // (Note the extensive use of matrix multiplications)
  // 
  //   m = 6 # input size
  //   n = 8 # output size, i.e. LSTM hidden size (in Keras)
  //   b = 4 # batch size (or timesteps)
  //   k = 2 # num iterations, i.e. n_steps
  //   x = np.random.randn(m, b)
  //   u = np.random.randn(m, k)
  //   v = np.random.randn(n, k)
  //   s = np.random.randn(k)
  //   s2 = np.random.randn(k)
  //   
  //   s_mat = np.repeat(s, m).reshape(k, m).T
  //   us = u * s_mat
  //   ux = x.T @ us
  //   y_batched = ux @ v.T
  // ===========================================================================
  // ===========================================================================
  // Define: x.T @ US Matrixes
  // ===========================================================================
  // Current x
  float *cur_i_ux = new float[num_inputs * num_timesteps * n_steps];
  float *cur_f_ux = new float[num_inputs * num_timesteps * n_steps];
  float *cur_c_ux = new float[num_inputs * num_timesteps * n_steps];
  float *cur_o_ux = new float[num_inputs * num_timesteps * n_steps];
  // Recurrent x
  float *rec_i_uh = new float[num_inputs * n_steps];
  float *rec_f_uh = new float[num_inputs * n_steps];
  float *rec_c_uh = new float[num_inputs * n_steps];
  float *rec_o_uh = new float[num_inputs * n_steps];

  // Current y
  float *cur_i_y = new float[num_inputs * num_timesteps * hidden_size];
  float *cur_f_y = new float[num_inputs * num_timesteps * hidden_size];
  float *cur_c_y = new float[num_inputs * num_timesteps * hidden_size];
  float *cur_o_y = new float[num_inputs * num_timesteps * hidden_size];
  // Recurrent y
  float *rec_i_y = new float[num_inputs * hidden_size];
  float *rec_f_y = new float[num_inputs * hidden_size];
  float *rec_c_y = new float[num_inputs * hidden_size];
  float *rec_o_y = new float[num_inputs * hidden_size];

  // ===========================================================================
  // NOTE: Broadcasting bias vector into a matrix for exploiting C initialization
  // in gemm function (C = A * B + C) produces no significant gain in terms of
  // execution time and makes the code harder to read...
  // ===========================================================================
  // Output y + bias
  float *i_cur_bias = new float[num_inputs * num_timesteps * hidden_size];
  float *f_cur_bias = new float[num_inputs * num_timesteps * hidden_size];
  float *c_cur_bias = new float[num_inputs * num_timesteps * hidden_size];
  float *o_cur_bias = new float[num_inputs * num_timesteps * hidden_size];

  float *i_sum = new float[num_inputs * hidden_size];
  float *f_sum = new float[num_inputs * hidden_size];
  float *c_sum = new float[num_inputs * hidden_size];
  float *o_sum = new float[num_inputs * hidden_size];

  float *i_gate = new float[num_inputs * hidden_size];
  float *f_gate = new float[num_inputs * hidden_size];
  float *o_gate = new float[num_inputs * hidden_size];
  float *c_sum_tanh = new float[num_inputs * hidden_size];
  float *c_tanh = new float[num_inputs * hidden_size];
  float *c_lhs = new float[num_inputs * hidden_size];
  float *c_rhs = new float[num_inputs * hidden_size];
  float *c = new float[num_inputs * hidden_size];

  // Recurrent v Transposed
  float *rec_i_v_T = new float[num_inputs *  hidden_size * n_steps];
  float *rec_f_v_T = new float[num_inputs *  hidden_size * n_steps];
  float *rec_c_v_T = new float[num_inputs *  hidden_size * n_steps];
  float *rec_o_v_T = new float[num_inputs *  hidden_size * n_steps];
  // ===========================================================================
  // Compute:
  // - S * U Matrix
  // ===========================================================================
  // Current us1
  float *cur_i_u_T = new float[input_size * n_steps];
  float *cur_f_u_T = new float[input_size * n_steps];
  float *cur_c_u_T = new float[input_size * n_steps];
  float *cur_o_u_T = new float[input_size * n_steps];
  // Recurrent us
  float *rec_i_u_T = new float[hidden_size * n_steps];
  float *rec_f_u_T = new float[hidden_size * n_steps];
  float *rec_c_u_T = new float[hidden_size * n_steps];
  float *rec_o_u_T = new float[hidden_size * n_steps];
  // Current us1
  float *cur_i_us = new float[num_inputs * n_steps * input_size];
  float *cur_f_us = new float[num_inputs * n_steps * input_size];
  float *cur_c_us = new float[num_inputs * n_steps * input_size];
  float *cur_o_us = new float[num_inputs * n_steps * input_size];
  // Recurrent us
  float *rec_i_us = new float[num_inputs * n_steps * hidden_size];
  float *rec_f_us = new float[num_inputs * n_steps * hidden_size];
  float *rec_c_us = new float[num_inputs * n_steps * hidden_size];
  float *rec_o_us = new float[num_inputs * n_steps * hidden_size];

  // ===========================================================================
  // NOTE: We need to 'transpose' u in order to matrix multiply it with x.
  // ===========================================================================
  // BEFORE TRANSPOSE: u.shape = (n_steps, input_size)
  svd_transpose(n_steps, input_size, cur_i_u, cur_i_u_T);
  svd_transpose(n_steps, input_size, cur_f_u, cur_f_u_T);
  svd_transpose(n_steps, input_size, cur_c_u, cur_c_u_T);
  svd_transpose(n_steps, input_size, cur_o_u, cur_o_u_T);
  // BEFORE TRANSPOSE: u.shape = (n_steps, hidden_size)
  svd_transpose(n_steps, hidden_size, rec_i_u, rec_i_u_T);
  svd_transpose(n_steps, hidden_size, rec_f_u, rec_f_u_T);
  svd_transpose(n_steps, hidden_size, rec_c_u, rec_c_u_T);
  svd_transpose(n_steps, hidden_size, rec_o_u, rec_o_u_T);

#ifdef USE_BLAS
  const bool kAvailableBLAS = true;
#else
  const bool kAvailableBLAS = false;
#endif

#ifdef SDS_DESIGN
  perf_counter sw_ctr;
  perf_counter gemm_ctr;
  perf_counter gemv_ctr;
  sw_ctr.start();
#elif !defined(__SYNTHESIS__)
  auto begin = std::chrono::high_resolution_clock::now();
#endif

  if (use_blas && kAvailableBLAS) {
#ifdef USE_BLAS
    for (int i = 0; i < num_samples; ++i) {
#ifdef SDS_DESIGN
      gemm_ctr.start();
#endif

      const int x_idx = i * num_inputs * input_size * num_timesteps;
      // Current x @ u.T: (num_timesteps * num_inputs, input_size) @ (n_steps, input_size).T
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps * num_inputs, n_steps, input_size, (float)1., &x[x_idx], cur_i_u_T, (float)0., cur_i_ux);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps * num_inputs, n_steps, input_size, (float)1., &x[x_idx], cur_f_u_T, (float)0., cur_f_ux);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps * num_inputs, n_steps, input_size, (float)1., &x[x_idx], cur_c_u_T, (float)0., cur_c_ux);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps * num_inputs, n_steps, input_size, (float)1., &x[x_idx], cur_o_u_T, (float)0., cur_o_ux);

      // Current ux * s: (num_timesteps * num_inputs, n_steps) * (n_steps,)
      for (int j = 0; j < num_timesteps; ++j) {
        for (int k = 0; k < num_inputs; ++k) {
          const int ux_idx = j * n_steps * num_inputs + k * n_steps;
          const int s_idx = k * n_steps;
          svd_mul(n_steps, &cur_i_ux[ux_idx], &cur_i_s[s_idx], &cur_i_us[ux_idx]);
          svd_mul(n_steps, &cur_f_ux[ux_idx], &cur_f_s[s_idx], &cur_f_us[ux_idx]);
          svd_mul(n_steps, &cur_c_ux[ux_idx], &cur_c_s[s_idx], &cur_c_us[ux_idx]);
          svd_mul(n_steps, &cur_o_ux[ux_idx], &cur_o_s[s_idx], &cur_o_us[ux_idx]);
        }
      }

      // Current uxs @ v: (num_timesteps * num_inputs, n_steps) @ (n_steps, hidden_size) = (num_timesteps * num_inputs, hidden_size)
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps * num_inputs, hidden_size, n_steps, (float)1., cur_i_us, cur_i_v, (float)0., cur_i_y);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps * num_inputs, hidden_size, n_steps, (float)1., cur_f_us, cur_f_v, (float)0., cur_f_y);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps * num_inputs, hidden_size, n_steps, (float)1., cur_c_us, cur_c_v, (float)0., cur_c_y);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps * num_inputs, hidden_size, n_steps, (float)1., cur_o_us, cur_o_v, (float)0., cur_o_y);

#ifdef SDS_DESIGN
      gemm_ctr.stop();
#endif

      const int h_idx = i * num_inputs * hidden_size;
      svd_set(num_inputs * hidden_size, (float)0., c);
      svd_set(num_inputs * hidden_size, (float)0., &out[h_idx]);

      for (int j = 0; j < num_timesteps; ++j) {
#ifdef SDS_DESIGN
        gemv_ctr.start();
#endif
        // Orig: Recurrent u.T @ h: (hidden_size, n_steps).T @ (hidden_size, num_inputs)
        // Recurrent h @ u.T:(num_inputs, hidden_size) @ (n_steps, hidden_size).T = (num_inputs, n_steps)
        svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_inputs, n_steps, hidden_size, (float)1., &out[h_idx], rec_i_u_T, (float)0., rec_i_uh);
        svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_inputs, n_steps, hidden_size, (float)1., &out[h_idx], rec_f_u_T, (float)0., rec_f_uh);
        svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_inputs, n_steps, hidden_size, (float)1., &out[h_idx], rec_c_u_T, (float)0., rec_c_uh);
        svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_inputs, n_steps, hidden_size, (float)1., &out[h_idx], rec_o_u_T, (float)0., rec_o_uh);

        // Recurrent hu * s: (num_inputs, n_steps) * (n_steps,)
        for (int k = 0; k < num_inputs; ++k) {
          const int s_idx = k * n_steps;
          svd_mul(n_steps, &rec_i_uh[s_idx], &rec_i_s[s_idx], &rec_i_us[s_idx]);
          svd_mul(n_steps, &rec_f_uh[s_idx], &rec_f_s[s_idx], &rec_f_us[s_idx]);
          svd_mul(n_steps, &rec_c_uh[s_idx], &rec_c_s[s_idx], &rec_c_us[s_idx]);
          svd_mul(n_steps, &rec_o_uh[s_idx], &rec_o_s[s_idx], &rec_o_us[s_idx]);
        }

        // Orig: Recurrent v.T @ uh: (n_steps, hidden_size).T @ (n_steps, num_inputs)
        // Recurrent uhs @ v: (num_inputs, n_steps) @ (n_steps, hidden_size) = (num_inputs, hidden_size)
        svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_inputs, hidden_size, n_steps, (float)1., rec_i_us, rec_i_v, (float)0., rec_i_y);
        svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_inputs, hidden_size, n_steps, (float)1., rec_f_us, rec_f_v, (float)0., rec_f_y);
        svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_inputs, hidden_size, n_steps, (float)1., rec_c_us, rec_c_v, (float)0., rec_c_y);
        svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_inputs, hidden_size, n_steps, (float)1., rec_o_us, rec_o_v, (float)0., rec_o_y);

#ifdef SDS_DESIGN
        gemv_ctr.stop();
#endif
        // =======================================================================
        // Non Linearities
        // =======================================================================
        const int x_cur_idx = j * num_inputs * hidden_size;
        const int h_size = num_inputs * hidden_size;
        svd_add(h_size, &cur_i_y[x_cur_idx], bias_i, i_cur_bias);
        svd_add(h_size, &cur_f_y[x_cur_idx], bias_f, f_cur_bias);
        svd_add(h_size, &cur_c_y[x_cur_idx], bias_c, c_cur_bias);
        svd_add(h_size, &cur_o_y[x_cur_idx], bias_o, o_cur_bias);

        svd_add(h_size, i_cur_bias, rec_i_y, i_sum);
        svd_add(h_size, f_cur_bias, rec_f_y, f_sum);
        svd_add(h_size, c_cur_bias, rec_c_y, c_sum);
        svd_add(h_size, o_cur_bias, rec_o_y, o_sum);

        svd_hard_sigmoid(h_size, i_sum, i_gate);
        svd_hard_sigmoid(h_size, f_sum, f_gate);
        svd_hard_sigmoid(h_size, o_sum, o_gate);
        svd_tanh(h_size, c_sum, c_sum_tanh);
        svd_mul(h_size, c_sum_tanh, i_gate, c_lhs);
        svd_mul(h_size, c, f_gate, c_rhs);

        svd_add(h_size, c_lhs, c_rhs, c);
        svd_tanh(h_size, c, c_tanh);
        svd_mul(h_size, c_tanh, o_gate, &out[h_idx]);
      }
    } 
#endif // end USE_BLAS
  } else {
//     // =========================================================================
//     // @todo (11/02/2019): openmp NOT WORKING: threads still share resources,
//     // creating data races and producing incorrect results.
//     // =========================================================================
//     // #pragma omp parallel for schedule(static, 8)
//     // =========================================================================
// // #pragma omp parallel for num_threads(8) schedule(dynamic)
//     for (int i = 0; i < num_samples; ++i) {
//       hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_i_us, cur_i_ux);
//       hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_f_us, cur_f_ux);
//       hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_c_us, cur_c_ux);
//       hls_gemm<float, float>(num_timesteps, n_steps, input_size, &x[i*input_size*num_timesteps], cur_o_us, cur_o_ux);

//       hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_i_ux, cur_i_v, cur_i_y);
//       hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_f_ux, cur_f_v, cur_f_y);
//       hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_c_ux, cur_c_v, cur_c_y);
//       hls_gemm<float, float>(num_timesteps, hidden_size, n_steps, cur_o_ux, cur_o_v, cur_o_y);

//       svd_set(hidden_size, (float)0., c);
//       svd_set(hidden_size, (float)0., &out[i * hidden_size]);

//       for (int j = 0; j < num_timesteps; ++j) {
//         hls_gemv<float, float>(n_steps, hidden_size, rec_i_u_T, &out[i*hidden_size], rec_i_uh);
//         hls_gemv<float, float>(n_steps, hidden_size, rec_f_u_T, &out[i*hidden_size], rec_f_uh);
//         hls_gemv<float, float>(n_steps, hidden_size, rec_c_u_T, &out[i*hidden_size], rec_c_uh);
//         hls_gemv<float, float>(n_steps, hidden_size, rec_o_u_T, &out[i*hidden_size], rec_o_uh);

//         hls_gemv<float, float>(hidden_size, n_steps, rec_i_v_T, rec_i_uh, rec_i_y);
//         hls_gemv<float, float>(hidden_size, n_steps, rec_f_v_T, rec_f_uh, rec_f_y);
//         hls_gemv<float, float>(hidden_size, n_steps, rec_c_v_T, rec_c_uh, rec_c_y);
//         hls_gemv<float, float>(hidden_size, n_steps, rec_o_v_T, rec_o_uh, rec_o_y);
//         // =======================================================================
//         // Non Linearities
//         // =======================================================================
//         svd_add(hidden_size, &cur_i_y[j * hidden_size], bias_i, i_cur_bias);
//         svd_add(hidden_size, &cur_f_y[j * hidden_size], bias_f, f_cur_bias);
//         svd_add(hidden_size, &cur_c_y[j * hidden_size], bias_c, c_cur_bias);
//         svd_add(hidden_size, &cur_o_y[j * hidden_size], bias_o, o_cur_bias);

//         svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
//         svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
//         svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
//         svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

//         svd_hard_sigmoid(hidden_size, i_sum, i_gate);
//         svd_hard_sigmoid(hidden_size, f_sum, f_gate);
//         svd_hard_sigmoid(hidden_size, o_sum, o_gate);
//         svd_tanh(hidden_size, c_sum, c_sum_tanh);
//         svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
//         svd_mul(hidden_size, c, f_gate, c_rhs);

//         svd_add(hidden_size, c_lhs, c_rhs, c);
//         svd_tanh(hidden_size, c, c_tanh);
//         svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
//       }
//     }
  }
  // ===========================================================================
  // NOTE: We are NOT taking into account the time it takes to both setup the u @ s
  // matrices and perform their transpositions because these operations can be
  // done "offline", i.e. can be stored in that form already, performance-wise.
  // ===========================================================================
#ifdef SDS_DESIGN
  sw_ctr.stop();
#elif !defined(__SYNTHESIS__)
  auto end = std::chrono::high_resolution_clock::now();
#endif

  if (verbose == 1) {
#ifdef SDS_DESIGN
    auto sw_cycles = sw_ctr.avg_cpu_cycles();
    auto sw_freq = sds_clock_frequency();
    std::cout << "Frequency: " << sw_freq << " ticks/second\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[SW-2-LSTM - BLAS version] ";
    } else {
      std::cout << "[SW-2-LSTM - no BLAS] ";
    }
    std::cout << "Batched SVD: Total CPU cycles: " << std::setprecision(12)
              << sw_cycles << std::setprecision(6) << "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[SW-2-LSTM - BLAS version] ";
    } else {
      std::cout << "[SW-2-LSTM - no BLAS] ";
    }
    std::cout << "Batched SVD: Average CPU cycles per sample: "
              << sw_cycles / num_samples << "\n";
    std::cout << "Batched SVD: Average CPU cycles per timesteps: "
              << sw_cycles / num_samples / num_timesteps << "\n";


    auto gemm_cycles = gemm_ctr.avg_cpu_cycles();
    if (use_blas && kAvailableBLAS) {
      std::cout << "[SW-2-LSTM - BLAS version] ";
    } else {
      std::cout << "[SW-2-LSTM - no BLAS] ";
    }
    std::cout << "Batched SVD-GEMM: Total CPU cycles: " << std::setprecision(12)
              << gemm_cycles << std::setprecision(6) << "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[SW-2-LSTM - BLAS version] ";
    } else {
      std::cout << "[SW-2-LSTM - no BLAS] ";
    }
    std::cout << "Batched SVD-GEMM: Average CPU cycles per sample: "
              << gemm_cycles / num_samples << "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[SW-2-LSTM - BLAS version] ";
    } else {
      std::cout << "[SW-2-LSTM - no BLAS] ";
    }
    std::cout << "Batched SVD-GEMM: Average CPU cycles per timesteps: "
              << gemm_cycles / num_samples / num_timesteps << "\n";

    auto gemv_cycles = gemv_ctr.avg_cpu_cycles();
    if (use_blas && kAvailableBLAS) {
      std::cout << "[SW-2-LSTM - BLAS version] ";
    } else {
      std::cout << "[SW-2-LSTM - no BLAS] ";
    }
    std::cout << "Batched SVD-GEMV: Total CPU cycles: " << std::setprecision(12)
              << gemv_cycles << std::setprecision(6) << "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[SW-2-LSTM - BLAS version] ";
    } else {
      std::cout << "[SW-2-LSTM - no BLAS] ";
    }
    std::cout << "Batched SVD-GEMV: Average CPU cycles per sample: "
              << gemv_cycles / num_samples << "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[SW-2-LSTM - BLAS version] ";
    } else {
      std::cout << "[SW-2-LSTM - no BLAS] ";
    }
    std::cout << "Batched SVD-GEMV: Average CPU cycles per timesteps: "
              << gemv_cycles / num_samples / num_timesteps << "\n";

#elif !defined(__SYNTHESIS__)
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - 
      begin).count();
    auto duration_us = duration_ns / 1000.0;
    auto duration_ms = duration_us / 1000.0;
    auto duration_s = duration_ms / 1000.0;
    if (use_blas && kAvailableBLAS) {
      std::cout << "[SW-2-LSTM - BLAS version] ";
    } else {
      std::cout << "[SW-2-LSTM - no BLAS] ";
    }
    std::cout << "Batched SVD: Total time: " << duration_ms
              << " ms (" << duration_s << " s)."<< "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[SW-2-LSTM - BLAS version] ";
    } else {
      std::cout << "[SW-2-LSTM - no BLAS] ";
    }
    std::cout << "Batched SVD: Average time per sample: " << duration_ms / num_samples
              << " ms (" << duration_s / num_samples << " s)."<< "\n";
    if (use_blas && kAvailableBLAS) {
      std::cout << "[SW-2-LSTM - BLAS version] ";
    } else {
      std::cout << "[SW-2-LSTM - no BLAS] ";
    }
    std::cout << "Batched SVD: Average time per timesteps: " << duration_ms / num_samples / num_timesteps
              << " ms (" << duration_s / num_samples / num_timesteps << " s)."<< "\n";
#endif
  }
  delete[] cur_i_ux;
  delete[] cur_f_ux;
  delete[] cur_c_ux;
  delete[] cur_o_ux;
  delete[] rec_i_uh;
  delete[] rec_f_uh;
  delete[] rec_c_uh;
  delete[] rec_o_uh;
  delete[] cur_i_y;
  delete[] cur_f_y;
  delete[] cur_c_y;
  delete[] cur_o_y;
  delete[] rec_i_y;
  delete[] rec_f_y;
  delete[] rec_c_y;
  delete[] rec_o_y;
  delete[] i_cur_bias;
  delete[] f_cur_bias;
  delete[] c_cur_bias;
  delete[] o_cur_bias;
  delete[] i_sum;
  delete[] f_sum;
  delete[] c_sum;
  delete[] o_sum;
  delete[] i_gate;
  delete[] f_gate;
  delete[] o_gate;
  delete[] c_sum_tanh;
  delete[] c_tanh;
  delete[] c_lhs;
  delete[] c_rhs;
  delete[] c;
  delete[] rec_i_v_T;
  delete[] rec_f_v_T;
  delete[] rec_c_v_T;
  delete[] rec_o_v_T;
  delete[] cur_i_u_T;
  delete[] cur_f_u_T;
  delete[] cur_c_u_T;
  delete[] cur_o_u_T;
  delete[] rec_i_u_T;
  delete[] rec_f_u_T;
  delete[] rec_c_u_T;
  delete[] rec_o_u_T;
  delete[] cur_i_us;
  delete[] cur_f_us;
  delete[] cur_c_us;
  delete[] cur_o_us;
  delete[] rec_i_us;
  delete[] rec_f_us;
  delete[] rec_c_us;
  delete[] rec_o_us;
}
#endif
