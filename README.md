# Unsupervised Dynamic Sampling (UDS)

## Overview

Unsupervised Dynamic Image Sampler (UDS) is a Python package designed to handle dynamic sampling of images. It is useful for microscopy applications where the user wants to sample a small fraction of pixels from an image (e.g. Scanning Electron Microscopy, Fluorescence Microscopy). The main goal is to calculate the next batch of locations faster than it takes for the equipment to measure the previous batch. That can be anywhere from 1 us per pixel to 1 ms per pixel.

## Features

- Fast Dynamic sampling in linear time per batch with respect to the number of pixels.
- Support for CPU / GPU via PyTorch.
- Easy-to-use ask & tell interface.

## Installation

To install UDS, you can use pip by providing the URL for the package. Assuming the package is hosted on a Git repository, you can install it as follows:

```bash
pip install git+https://github.com/ZacharyVarley/uds.git
```


## Basic Usage

Here is a simple example that shows how you can use UDS:

```python
import torch
from uds.sampler import DynamicSampling

# specify the fraction to sample, the region of interest shape
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
region_shape = (100, 100) # region of interest shape
image = torch.rand(region_shape, device=device) # create a random image
var_wr = 2 # window radius on local variance calculation
arm_wr = 1 # window radius on argrelmax calculation
fraction = 0.1 # fraction of pixels to sample
var_thresh = 0.0005 # minimum for meaningful variance
total_pixels_to_sample = int(fraction * region_shape[0] * region_shape[1])

# Initialize the DynamicSampler object
sampler = DynamicSampling(device, region_shape, fraction, var_wr, arm_wr, variance_threshold)

# Loop and ask for pixel batches to sample
while sampler.num_sampled < total_pixels_to_sample:
    # Ask for a batch of pixels to sample
    coords = sampler.ask()
    # Tell the sampler the grayscale values of the pixels
    sampler.tell(image[coords[:, 0], coords[:, 1]])

# Retrieve the sampled pixels
sampled_mask = sampler.mask

```

## License

This project is licensed under a BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

---