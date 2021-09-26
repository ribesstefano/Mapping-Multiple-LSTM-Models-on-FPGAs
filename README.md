# HLS SVD

## Overview

This project includes the HLS implementation and the approximation algorithms included in the paper "Mapping Multiple LSTM models on FPGA".

### SVD Approximation Algorithm

The approximation algorithms are in the `python/` folder.

## Requirements

* CMake
* Xilinx Vivado 2018.3 (deprecated)
* Xilinx Vitis 2021.1 (deprecated)

### CMake Simulation

#### Windows

Simulation is working assuming the Xilinx OpenCV DLLs are copied into the `bin/` folder along side with the generated executables.
```
"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
mkdir build
cmake .. -G Ninja
cmake --build . --config Release
```

#### Linux
```
mkdir build
cd build
cmake ..
make all
```

## Notes on Using Vitis HLS

### AXIS Interface and DMA

Vitis will include the TLAST side channel if and only if TKEEP and TSTRB are also included.

In order to attach the port to a Xilinx DMA, the TLAST signal must be properly set HIGH at the end of the data transmission.

The TKEEP and TSTRB signals must be *always* set to HIGH, as indicated in the [AXIS documentation](https://developer.arm.com/documentation/ihi0051/a/Interface-Signals/Byte-qualifiers/TKEEP-and-TSTRB-combinations).

Note: for using external DMAs, we need the TLAST, TKEEP and TSTRB signals. In particular, TKEEP and TSTRB must be all set (i.e. all ones) in order to signal data packets.

#### AxiStreamInterface Class

This repository contains a wrapper class for kernel arguments of type `hls::stream` named `AxiStreamInterface`. The class is implemented following a _Policy-based_ C++ paradigm, meaning that it accepts either a `AxiStreamPort` or `AxiStreamFifo` as possible policies (in practice, a template argument).

The idea is to have a kernel argument, i.e. an HLS port, which can be either an AXIS interface with side-channels, or a bare FIFO interface connected to another kernel. In fact, Vitis HLS doesn't allow stream interfaces with side-channels within an IP. To overcome the issue, the `AxiStreamInterface` can be customized to be an IP port or a FIFO port, depending on the use of the kernel.

An example of this can be seen in `HlsKernelU` and in `svd::SvdKernel`, which specialize the `svd::KernelU` function template. In the first case, the `svd::KernelU` has its output stream port `xu_port` connected to one of the IP's ports (with side-channels). In the latter case instead, `svd::KernelU` is connected to `svd::KernelS`, and so its `xu_port` argument is an internal FIFO (without side-channels).

The `AxiStreamInterface` class in `axis_lib.h` can also be used with `hls::vector` types.

### AXIS Interfaces and `depth`

In order to implement AXIS interfaces, avoid using `depth` in the pragma, as follows:
```c++
const int kAxiBitwidth = 128;

void HlsVectorKernelU(hls::stream<ap_axiu<kAxiBitwidth, 0, 0, 0> >& x_port,
                      hls::stream<ap_axiu<kAxiBitwidth, 0, 0, 0> >& y_port) {
#pragma HLS INTERFACE axis port=x_port // depth=... <- DON'T SPECIFY THE DEPTH!
#pragma HLS INTERFACE axis port=y_port // depth=... <- DON'T SPECIFY THE DEPTH!
	// ...
}
```
The type `ap_axiu` must now be used to generate AXIS with side channels.

## hls::vector Arrays on AXI-Lite Interfaces

In Vitis 2021.1 it **not** allowed to have `hls::vector` type arguments mapped to AXI-Lite interfaces.
Instead, use a bare arrays, *e.g.* `const int x[N]` instead of `const hls::vector<int, N> x`.


### Partitioning hls::vector Arrays

A standard way of partitioning an array is:
```c++
  hls::stream<hls::vector<int, 4> > x_streams[M][N];
#pragma HLS ARRAY_PARTITION variable=x_streams complete dim=0
```
However, since we are dealing with a `hls::vector` type, setting `dim=0` (all dimensions) will partition the array on the vector dimension too.

In the example above, Vitis will create `M * N * 4` different streams (instead of just `M * N`). To fix it, manually specify the partitioning on the dimensions, like so:
```c++
  hls::stream<hls::vector<int, 4> > x_streams[M][N];
#pragma HLS ARRAY_PARTITION variable=x_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=x_streams complete dim=2
```

### HLS Vector Patch

If the project will be compiled with the Vitis HLS libraries, it needs a patch in the `hls::vector` class.

Simply add the following line in the `vector` class after the `public:` statement:
```c++
public:
  static const int width = N;
```

In this way, one can access the number of elements in a `hls::vector` at compile/synthesis time by doing:

```c++
hls::vector<int, 5> a;
std::cout << "Number of elements in a: " << a::width << std::endl;

// > Number of elements in a: 5
```

## TODOs

List of TODOs:

	* Properly test the pruned versions of Kernel-U and Kernel-V.
	* ~Import u, s, v new kernels~

## Bugs

List of possible bugs:

* Constructing data handler storage might lead to segmentation faults.
* Having `num_active_inputs == 1` is breaking in hardware runs.