# SpaceFlow: Identifying Multicellular Spatiotemporal Organization of Cells using Spatial Transcriptome Data

SpaceFlow is Python package for identifying spatiotemporal patterns and spatial domains from Spatial Transcriptomic (ST) Data. Based on deep graph network, SpaceFlow provides the following functions:  
1. Encodes the ST data into **low-dimensional embeddings** that reflecting both expression similarity and the spatial proximity of cells in ST data.
2. Incorporates **spatiotemporal** relationships of cells or spots in ST data through a **pseudo-Spatiotemporal Map (pSM)** derived from the embeddings.
3. Identifies **spatial domains** with spatially-coherent expression patterns.

## Installation

### 1. Prepare an isolated package dependency environment (Optional)
To install SpaceFlow, we recommend using the [Anaconda Python Distribution](https://anaconda.org/) and creating an environment, so the SpaceFlow code and dependencies don't interfere with anything else. Here is the command to create an environment:

```bash
conda create -n spaceflow_env python=3.7
```

After create the environment, you can activate the `spaceflow_env` environment by:
```bash
conda activate spaceflow_env
```

You can deactivate the `spaceflow_env` environment by:
```bash
conda deactivate
```

### 2. Install dependencies
SpaceFlow depends on several packages, such as: [torch](https://pytorch.org/), [numpy](https://numpy.org/), [scipy](https://scipy.org/), [networkx](https://networkx.org/), etc. See a full list of the dependencies in `requirements.txt` file.

Before install the SpaceFlow, dependencies need be installed through:
```bash
pip install --user --requirement requirements.txt
```

### 3. Install SpaceFlow
Switch to the `spatial-constrained-clustering-and-pseudotime` directory of the downloaded SpaceFlow package in the terminal, and run the following command to install SpaceFlow:
```bash                                          
pip install --user .
```

## Usage

### A Quick Start Tutorial

We will use a mouse organogenesis ST data to demonstrate the usage of SpaceFlow. The ST data is generated through [seqFISH](https://spatial.caltech.edu/seqfish/) from [(Lohoff, T. et al. 2022)](https://doi.org/10.1038/s41587-021-01006-2) and is available in [squidpy](https://squidpy.readthedocs.io/en/stable/) package.







