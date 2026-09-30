# Multi-Band Blending

by Andre Biermeier

This package implements multi-band blending of two images using linear
feathering, cosine feathering or graph cut based masks.
It supports grayscale and RGB images, both with or without an alpha channel.

## Installation

Use Python 3.11 or newer. Installing from source requires a C++20 compiler
and CMake 3.25 or newer. NumPy and the Python build dependencies are installed
automatically.

Clone this repository, open its directory and install the package:

```bash
python -m pip install .
```

Alternatively, install directly from the Git repository:

```bash
python -m pip install "git+https://github.com/AndreBiermeier/Multi-Band-Blending.git@main"
```

This method also requires Git to be installed.

## Usage

This example creates two RGB images with a horizontal overlap of 48 pixels:

```python
import numpy as np
from multiband_blending import ImageOrigin, blend

image_a = np.full((64, 96, 3), [0.8, 0.3, 0.2], dtype=np.float32)
image_b = np.full((64, 96, 3), [0.2, 0.4, 0.8], dtype=np.float32)

result = blend(
    image_a,
    image_b,
    origin_a=ImageOrigin(y=0, x=0),
    origin_b=ImageOrigin(y=0, x=48),
    mask="graphcut",
)
print(result.blended_image.shape)  # (64, 144, 4)
```

Images must already be aligned. Origins specify integer pixel positions in
`(y, x)` order. Set `mask` to `"linear"`, `"cosine"`, or `"graphcut"`.
Graph cut requires a partial rectangular overlap at least two pixels wide and
high.

Inputs can be grayscale or RGB arrays with optional alpha. Both
images must use the same color format.

`result.blended_image` contains float32 values in `[0, 1]` and always includes
alpha: two channels for grayscale or four for RGB. `result.overlap_mask` contains
the overlap weights, where 1 selects image A and 0 selects image B.
`result.canvas_origin` gives the output's top-left position.

Optional arguments are `levels=5` for the maximum pyramid depth,
`kernel_size=5` for the pyramid filter width, and `mask_kernel_size=5` for
graph-cut mask smoothing. Kernel widths must be positive odd integers.
`levels=None` uses all available levels. `levels=1` performs direct weighted
blending. `mask_kernel_size` only affects graph cut.
