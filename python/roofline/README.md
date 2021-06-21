# Roofline Model

The maximum bandwidth can be obtained [here](https://www.xilinx.com/support/documentation/user_guides/ug585-Zynq-7000-TRM.pdf).

A discussion on the peak floating point performace can be obtained [here](https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/wp/wp-01222-understanding-peak-floating-point-performance-claims.pdf).

## Hardware Design

## Number of Iterations

Increasing the number of iterations shouldn't signlificantly increase the resource usage.
The amount of iterations may heavily impact on the DMAs.

## Dot Product Unit

It consumes most of the DSPs.

## Non Linear Unit

This block is the most resource hungry, so parallelizing it might be expensive.

## Weight Memory Layout

The weight matrixes of two LSTM layers are divided in 8 gates: 4 current and 4 recurrent.
Each gate matrix is 'shared' by the two layers thanks to the modified SVD algorithm.
Because of the algorithm, each gate matrix is stored in different vectors, i.e. `U`, `V`, `S1` and `S2`.
Each of these vectors is, in reality, a *tensor*, meaning that the hardware accelerator assumes a certain tensor shape in order to properly fetch the data and improve throughput.

The tensors' shapes are:

* `V`: (I, G, T, E)
* `U`: (I, G, T, E)
* `S1` + `S2`: (I, S, G, E)

with:

* I: number of iterations or refinements
* G: number of LSTM gates
* T: number of **non zero** tiles
* E: number of elements (the actual weight values)

Note that `S1` and `S2` are stored 'contiguously' in the **same tensor**.

### Zero Tiles Combinations: Layout

The accelerator reads a set of indexes indicating which tiles have been pruned.
These combinations are stored in the following layout:

* `VZ`: (I, G, T)
* `UZ`: (I, G, T)

Where T is a bit vector of width `NumTiles`, with coding: `1` non-pruned tile, `0` pruned tile.

**Assumption**: both current and recurrent gates share the same number of tiles and number of zero tiles, despite of having **different matrix dimensions**.

## Points to Discuss / TODOs

### Roofline Model

* Which peak FLOP performance shall we use?
* The total number of operations: is it referred to the original LSTM algorithm or the SVD-based one?
* In Computational Roof calculation, how do I decide the slowest module?
* In Computational Roof calculation, is MAC flop equal to MUL flop, i.e. 1? Or `MAC = ADD + MUL = 2`

### Hardware Accelerator

* The current and recurrent gates have different matrix dimensions!
* The previous hidden state `c` can, in principle, be stored on the device for reuse across timesteps.
* If the LSTM is not required to **return a sequence**, the output writes to memory can be avoided and a set of buffers can be used instead.
* If the RTL/Cosimulation succeeds, then there is no need to run the hardware accelerator for testing the accuracy (for performance measurements instead, we obviously need the hardware)
* Both the C and the RTL/Cosimulations are breaking when printing the `mean_error_value`. I believe it might be related to the `UpdateErrorMean()` function, which is modifying it.
