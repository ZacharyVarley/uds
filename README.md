# Unsupervised Dynamic Sampling (UDS)

## Overview

Unsupervised Dynamic Image Sampler (UDS) is a Python package designed to handle dynamic sampling of images without the need for labeled data. It is useful for researchers and developers working on image processing, machine learning, and computer vision projects where labeled data is not available.

## Features

- Fast Dynamic sampling based in linear time.
- Support for CPU / GPU via PyTorch.
- Easy-to-use ask / tell interface.

## Installation

To install UDS, you can use pip by providing the URL for the package. Assuming the package is hosted on a Git repository, you can install it as follows:

```bash
pip install git+https://github.com/ZacharyVarley/uds.git
```


## Basic Usage

Here is a simple example that shows how you can use UDS:

```python
import torch
from uds import DynamicSampler

# specify the fraction to sample, the region of interest shape
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
region_shape = (100, 100)
image = torch.rand(region_shape)
fraction = 0.1
variance_threshold = 0.0005 # minimum for meaningful variance

# Initialize the DynamicSampler object
sampler = DynamicSampler(
```

## License

This project is licensed under a BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

---