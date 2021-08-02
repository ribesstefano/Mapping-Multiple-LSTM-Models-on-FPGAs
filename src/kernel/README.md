# Kernels

## U-Kernel

### HlsAxisKernelU

To be used with external DMAs.
```c++
void HlsAxisKernelU(const int num_refinements,
  hls::stream<typename testu::VectTuAxiType>& x_port,
  hls::stream<typename testu::VectTuAxiType>& u_port,
  hls::stream<typename testu::VectGN_AxiType>& xu_port);
```

### HlsManySamplingsKernelU

Compared to the previous implementation, this kernel has a different number of refinements per input. The refinements and inputs must be **ordered**. Meaning that input at index zero has the lowest amount of refinements to process.

```c++
void HlsManySamplingsKernelU(const hls::vector<int, testu::params::N> num_refinements,
  hls::stream<typename testu::VectTuAxiType>& x_port,
  hls::stream<typename testu::VectTuAxiType>& u_port,
  hls::stream<typename testu::VectN_AxiType>& xu_port);
```