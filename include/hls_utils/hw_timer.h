#ifndef HLS_UTILS_HW_TIMER_H_
#define HLS_UTILS_HW_TIMER_H_

#include "hls_stream.h"

namespace hlsutils {

/**
 * @brief      Synthesizeable HLS clock counter. Note: this module must be
 *             placed in a DATAFLOW region and at the bottom of it (to avoid
 *             leftovers in the control fifos). The function to be measured will
 *             control the counter.
 *
 * @param      probe_ctrl    The probe control interface: start the counter by
 *                           sending a 1: probe_ctrl.write(1); and stop it bu
 *                           sending a 0: probe_ctrl.write(0);
 * @param      stop_ctrl     The stop control interface: the counter must be
 *                           switched off for avoiding infinite-loop hangs in
 *                           synthesis. In order to do so, send a 1 to this
 *                           interface: stop_ctrl.write(1);
 * @param      counter_port  The counter value to be reported
 *
 * @tparam     T             The counter type, e.g. unsigned long long,
 *                           ap_uint<64>, int, unsigned, etc...
 */
template <typename T = unsigned long long>
void ClockCounter(hls::stream<bool> &probe_ctrl, hls::stream<bool> &stop_ctrl,
    T &counter_port) {
#ifdef __SYNTHESIS__
  // ===========================================================================
  // NOTE: The key idea is in the read_nb() method: in software, it will always
  // return true and set the read value to 0 in case of an empty fifo (which
  // will always be, in software). In hardware instead, read_nb() will evaluate
  // false most of the time, until there actually is a value in the fifo to
  // read, thus preventing the while loop from exiting.
  // ===========================================================================
  T clk_count = 0;
  T probe_start = 0;
  bool stop_module_signal = 0;

  while(stop_module_signal == 0) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
#pragma HLS PIPELINE II=1
    // =========================================================================
    // Start and stop of the clock counter.
    // =========================================================================
    bool probe_data = 1;
    if (probe_ctrl.read_nb(probe_data) == 1) {
      if (probe_data == 0) {
        counter_port += clk_count - probe_start; // Record counter end.
      } else {
        probe_start = clk_count; // Record counter start.
      }
    }
    // =========================================================================
    // Counter termination.
    // =========================================================================
    bool stop_signal = 1;
    if (stop_ctrl.read_nb(stop_signal) == 1) {
      if (stop_signal == 0) {
        stop_module_signal = 1;
      }
    }
    // NOTE: "clk_count" becomes the actual HW cycle + 1. This is possible
    // because this while loop achieves II=1.
    clk_count += 1;
  }
#else
  // In software mode, empty the fifos for avoiding leaving leftovers.
  while(!probe_ctrl.empty()) {
    auto dummy_read = probe_ctrl.read();
  }
  while(!stop_ctrl.empty()) {
    auto dummy_read = stop_ctrl.read();
  }
#endif
}

template <typename T = unsigned long long>
void ClockCounter(hls::stream<bool> &probe_ctrl, hls::stream<bool> &stop_ctrl,
    T &counter_port, T &clk_count_port) {
#ifdef __SYNTHESIS__
  // ===========================================================================
  // NOTE: The key idea is in the read_nb() method: in software, it will always
  // return true and set the read value to 0 in case of an empty fifo (which
  // will always be, in software). In hardware instead, read_nb() will evaluate
  // false most of the time, until there actually is a value in the fifo to
  // read, thus preventing the while loop from exiting.
  // ===========================================================================
  T clk_count = 0;
  T probe_start = 0;
  bool stop_module_signal = 0;

  while(stop_module_signal == 0) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
#pragma HLS PIPELINE II=1
    // =========================================================================
    // Start and stop of the clock counter.
    // =========================================================================
    bool probe_data = 1;
    if (probe_ctrl.read_nb(probe_data) == 1) {
      if (probe_data == 0) {
        counter_port += clk_count - probe_start; // Record counter end.
      } else {
        probe_start = clk_count; // Record counter start.
      }
    }
    // =========================================================================
    // Counter termination.
    // =========================================================================
    bool stop_signal = 1;
    if (stop_ctrl.read_nb(stop_signal) == 1) {
      if (stop_signal == 0) {
        stop_module_signal = 1;
      }
    }
    // NOTE: "clk_count" becomes the actual HW cycle + 1. This is possible
    // because this while loop achieves II=1.
    clk_count += 1;
  }
  // Set the total clock time.
  clk_count_port = clk_count;
#else
  // In software mode, empty the fifos for avoiding leaving leftovers.
  while(!probe_ctrl.empty()) {
    auto dummy_read = probe_ctrl.read();
  }
  while(!stop_ctrl.empty()) {
    auto dummy_read = stop_ctrl.read();
  }
#endif
}


/**
 * @brief      Synthesizeable HLS clock counter. Note: this module must be
 *             placed in a DATAFLOW region and at the bottom of it (to avoid
 *             leftovers in the control fifos). The function to be measured will
 *             control the counter.
 *
 * @param      probe_ctrl      The probe control interface: start the counter by
 *                             sending a 1: probe_ctrl.write(1); and stop it by
 *                             sending a 0: probe_ctrl.write(0);
 * @param      stop_ctrl       The stop control interface: the counter must be
 *                             switched off for avoiding infinite-loop hangs in
 *                             synthesis. In order to do so, send a 1 to this
 *                             interface: stop_ctrl.write(1);
 * @param      counter_port    The counter value to be reported
 * @param      clk_count_port  The total clock count
 *
 * @tparam     T               The counter type, e.g. unsigned long long,
 *                             ap_uint<64>, int, unsigned, etc...
 * @tparam     NumProbes       Number of probes, i.e. counters, to use.
 */
template <typename T = unsigned long long, int NumProbes = 1>
void ClockCounter(hls::stream<bool> (&probe_ctrl)[NumProbes],
    hls::stream<bool> &stop_ctrl, T counter_port[NumProbes], T &clk_count_port) {
#ifdef __SYNTHESIS__
  // ===========================================================================
  // NOTE: The key idea is in the read_nb() method: in software, it will always
  // return true and set the read value to 0 in case of an empty fifo (which
  // will always be, in software). In hardware instead, read_nb() will evaluate
  // false most of the time, until there actually is a value in the fifo to
  // read, thus preventing the while loop from exiting.
  // ===========================================================================
  bool stop_module_signal = 0;
  bool probe_started[NumProbes];
  T internal_counters[NumProbes];
  T clk_count[NumProbes + 1]; // The last counter is used to report the total cycles
  T probe_start[NumProbes];
#pragma HLS ARRAY_PARTITION variable=clk_count complete
#pragma HLS ARRAY_PARTITION variable=probe_start complete

  Init: for (int i = 0; i < NumProbes; ++i) {
#pragma HLS UNROLL
    internal_counters[i] = 0;
    clk_count[i] = 0;
    probe_start[i] = 0; // otherwise it might not be properly reset accross calls
    probe_started[i] = false;
  }
  clk_count[NumProbes] = 0;

  while(stop_module_signal == 0) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
#pragma HLS PIPELINE II=1
    // =========================================================================
    // Start and stop of the clock counter.
    // =========================================================================
    for (int i = 0; i < NumProbes; ++i) {
      bool probe_data = 1;
      if (probe_ctrl[i].read_nb(probe_data) == 1) {
        if (probe_data == 0) {
          internal_counters[i] += clk_count[i] - probe_start[i]; // Record counter end.
          probe_started[i] = false;
        } else {
          if (!probe_started[i]) {
            probe_start[i] = clk_count[i]; // Record counter start.
            probe_started[i] = true;
          }
        }
      }
    }
    // =========================================================================
    // Counter termination.
    // =========================================================================
    bool stop_signal = 1;
    if (stop_ctrl.read_nb(stop_signal) == 1) {
      if (stop_signal == 0) {
        stop_module_signal = 1;
      }
    }
    // NOTE: "clk_count" becomes the actual HW cycle + 1. This is possible
    // because this while loop achieves II=1.
    for (int i = 0; i < NumProbes + 1; ++i) {
      clk_count[i] += 1;
    }
  }
  for (int i = 0; i < NumProbes; ++i) {
#pragma HLS UNROLL
#pragma HLS PIPELINE II=1
    auto leftover_time = clk_count[i] - probe_start[i];
    if (probe_started[i]) { // if we haven't received the stop, record the final timestamp
      counter_port[i] = internal_counters[i] + leftover_time;
    } else {
      counter_port[i] = internal_counters[i];
    }
  }
  clk_count_port = clk_count[NumProbes]; // The last counter is used to report the total cycles
#else
  // In software mode, empty the fifos for avoiding leaving leftovers.
  for (int i = 0; i < NumProbes; ++i) {
    while(!probe_ctrl[i].empty()) {
      auto dummy_read = probe_ctrl[i].read();
    }
  }
  while(!stop_ctrl.empty()) {
    auto dummy_read = stop_ctrl.read();
  }
#endif
}


/**
 * @brief      Synthesizeable HLS clock counter. Note: this module must be
 *             placed in a DATAFLOW region and controlled by the function being
 *             measured.
 *
 * @param      probe_ctrl           The probe control interface: start the
 *                                  counter by sending a 1: probe_ctrl.write(1);
 *                                  and stop it by sending a 0:
 *                                  probe_ctrl.write(0);
 * @param      stop_ctrl            The stop control interface: the counter must
 *                                  be switched off for avoiding infinite-loop
 *                                  hangs in synthesis. In order to do so, send
 *                                  a 1 to this interface: stop_ctrl.write(1);
 * @param      counter_port         The counter value to be reported
 * @param      counter_port_output  The counter value output to be reported in a
 *                                  stream. This will allow the controlling
 *                                  module to read the counter value, not only
 *                                  the top level function instantiating the
 *                                  counter.
 *
 * @tparam     T                    The counter type, e.g. unsigned long long,
 *                                  ap_uint<64>, int, unsigned, etc...
 */
template <typename T = unsigned long long>
void ClockCounter(hls::stream<bool> &probe_ctrl, hls::stream<bool> &stop_ctrl,
    T &counter_port, hls::stream<T> &counter_port_output) {
#ifdef __SYNTHESIS__
  T clk_count = 0;
  T probe_start = 0;
  bool stop_module_signal = 0;

  while(stop_module_signal == 0) {
#pragma HLS PIPELINE II=1
    // =========================================================================
    // Start and stop of the clock counter.
    // =========================================================================
    bool probe_data = 1;
    if (probe_ctrl.read_nb(probe_data) == 1) {
      if (probe_data == 0) {
        counter_port += clk_count - probe_start; // Record counter end.
        counter_port_output.write(counter_port);
      } else {
        probe_start = clk_count; // Record counter start.
      }
    }
    // =========================================================================
    // Counter termination.
    // =========================================================================
    bool stop_signal = 1;
    if (stop_ctrl.read_nb(stop_signal) == 1) {
      if (stop_signal == 0) {
        stop_module_signal = 1;
      }
    }
    // NOTE: "clk_count" becomes the actual HW cycle + 1. This is possible
    // because this while loop achieves II=1.
    clk_count += 1;
  }
#endif
}

/**
 * @brief      Internal hardware clock counter. It returns the enlapsed clock
 *             cycles from when it was started to when it was stopped.
 *             This module must be placed at the end of a dataflow region.
 *
 * @param      stop_ctrl  The stop control signal: it's actually a FIFO, send 1
 *                        to stop the module. It must be stopped to avoid
 *                        deadlocks
 *
 * @tparam     T          The clock counter type. Default: unsigned long long
 *
 * @return     The enlapsed number of clock cycles.
 */
template <typename T = unsigned long long>
void UpClockCounter(hls::stream<bool> &stop_ctrl, T &timestamp) {
#ifdef __SYNTHESIS__
  T clk_count = 0;
  bool keep_counting = true;
  bool stop_signal = 0;

  while(keep_counting) {
#pragma HLS PIPELINE II=1
    // =========================================================================
    // Check whether to stop the clock counter.
    // =========================================================================
    if (stop_ctrl.read_nb(stop_signal)) {
      if (stop_signal == 1) {
        timestamp = clk_count;
        keep_counting = false;
      }
    }
    // NOTE: "clk_count" becomes the actual HW cycle + 1. This is possible
    // because this while-loop achieves II=1.
    clk_count += 1;
  }
#else
  // In software mode, empty the fifos for avoiding leaving leftovers.
  while(!stop_ctrl.empty()) {
    auto dummy_read = stop_ctrl.read();
  }
#endif
}

const int NUM_HW_TIMERS = 1;

/*
 * The global variable hw_timers will be used when calling the hardware counter
 * functions. The idea is to have it globally, in order to have it accessible
 * from anywhere in the HLS design code.
 *
 * Note that global variables can be even mapped to the ports of the HLS IPs.
 * This could be useful to printout timing results in Cosimulation.
 */
static unsigned long long hw_timers[NUM_HW_TIMERS];

typedef long long CounterD;
typedef hls::stream<bool> ProbeStream;

} // hlsutils

#endif // end HLS_UTILS_HW_TIMER_H_