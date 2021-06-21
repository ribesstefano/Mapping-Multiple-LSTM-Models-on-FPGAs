# LSTM Software Models

This folder contains the software implementation and the hardware emulator of an LSTM layer.

## Software

The software version exploits the BLAS libraries for the fast computation of the gate matrix-matrix multiplications.

## Hardware

The hardware emulation functions serve as a mean to test the _accuracy_ of the HLS models. Because of that, the outputs produced by the emulator and the HLS must obviously match.

Compared to the HLS counterpart, the emulator:
	* utilizes dynamic parameters, meaning that they aren't statically defined at compile-time
	* has a software-friendly execution: the HLS coding style is in fact un-optimized for fast execution (e.g. the HLS has numerous if-statements inside loops)