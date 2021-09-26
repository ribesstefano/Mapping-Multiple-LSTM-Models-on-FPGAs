# Notes on PYNQ Designs

## Vivado Project

### Xilinx DMA

The DMA should be configured in the following way:

* Max burst length to maximum
* Register buffer width to maximum

### HP Ports

All HP ports should be set to their maximum size width (64bit for the PYNQ-Z1 board and 128bit for the ZCU104) in order to avoid receiving data interleaved by zeroes.

## Jupyter Notebook

### Generating Randomly-filled Buffer

```python
import numpy as np

R, N, G = 64, 2, 4

xus = np.random.randn(R, N, G).astype(dtype=np.int16)
xus_buffer = pynq.allocate(shape=(R, N, G), dtype=np.int16)
np.copyto(xus_buffer, xus, casting='no')
```

### Storing and Loading Weights from bin file

```python
import numpy as np

R, N, G = 64, 2, 4

tmp = np.random.randn(R, N, G).astype(dtype=np.int16)
tmp.tofile('binfile_example.bin')

def load_from_bin(binfile, shape, dtype):
    tmp_buffer = pynq.allocate(shape=shape, dtype=dtype)
    tmp = np.fromfile(binfile, dtype=data_t).reshape(tmp_buffer.shape)
    np.copyto(tmp_buffer, tmp, casting='no')
    return tmp_buffer

xus_buffer = load_from_bin('binfile_example.bin', shape=(R, N, G), dtype=np.int16)
```
