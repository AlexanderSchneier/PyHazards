# PyHazard

[![PyPI - Version](https://img.shields.io/pypi/v/PyHazard)](https://pypi.org/project/PyHazard)
[![Build Status](https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazard/docs.yml)](https://github.com/LabRAI/PyHazard/actions)
[![License](https://img.shields.io/github/license/LabRAI/PyHazard.svg)](https://github.com/LabRAI/PyHazard/blob/main/LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/PyHazard)](https://github.com/LabRAI/PyHazard)
[![Issues](https://img.shields.io/github/issues/LabRAI/PyHazard)](https://github.com/LabRAI/PyHazard)
[![Pull Requests](https://img.shields.io/github/issues-pr/LabRAI/PyHazard)](https://github.com/LabRAI/PyHazard)
[![Stars](https://img.shields.io/github/stars/LabRAI/PyHazard)](https://github.com/LabRAI/PyHazard)
[![GitHub forks](https://img.shields.io/github/forks/LabRAI/PyHazard)](https://github.com/LabRAI/PyHazard)

PyHazard is a Python library designed for using AI tools for hazard prediction. It provides a modular framework to implement and test prediction strategies on natural hazards.

## How to Cite

If you find it useful, please consider citing the following work:

```
```


## Installation

PyHazard supports both CPU and GPU environments. Make sure you have Python installed (version >= 3.8, <3.13).

### Base Installation

First, install the core package:

```bash
pip install PyHazard
```

This will install PyHazard with minimal dependencies.

### CPU Version

```bash
pip install "PyHazard[torch,dgl]" \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -f https://data.dgl.ai/wheels/repo.html
```

### GPU Version (CUDA 12.1)

```bash
pip install "PyHazard[torch,dgl]" \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple \
  -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```

## Quick Start

Here's a simple example to use PyHazard:

```python
from pyhazard.datasets import Cora
from pyhazard.models.attack import MEA

# Load dataset
dataset = Cora(api_type='pyg')

# Initialize and run Model Extraction Attack
attack = MEA(dataset=dataset)
attack.attack()
```

And a simple example to run a defense:

```python
from pyhazard.datasets import Cora
from pyhazard.models.defense import ATOM

# Load dataset
dataset = Cora(api_type='pyg')

# Initialize and run ATOM defense
defense = ATOM(dataset=dataset)
defense.defense()
```

If you want to use CUDA, please set environment variable:

```shell
export PYHAZARD_DEVICE=cuda:0
```

## Implementation & Contributors Guideline

Refer to [Implementation Guideline](.github/IMPLEMENTATION.md)

Refer to [Contributors Guideline](.github/CONTRIBUTING.md)

## License

[BSD 2-Clause License](LICENSE)

## Contact

For questions or contributions, please contact xc25@fsu.edu.
