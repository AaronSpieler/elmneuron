# Supplementary Material:
## The ELM Neuron: an Efficient and Expressive Cortical Neuron Model Can Solve Long-Horizon Tasks.

This repository features a minimal implementation of the ELM neuron, alongside a dataloader for the new SHD-Adding dataset. For training the ELM neuron on either the SHD or SHD-Adding dataset, please refer to the enclosed Jupyter Notebook.

### The following files are provided:
- __train_elm_on_shd.ipynb__: Notebook for training an ELM neuron on SHD or SHD-Adding Dataset.
- __expressive_leaky_memory_neuron.py__: Minimal implementation of the ELM neuron in PyTorch.
- __shd_download_utils.py__: Utilities for downloading the SHD Dataset.
- __shd_data_loader.py__: Data loaders for SHD and SHD-Adding.
- __elm_env.yml__: Definition of Conda environment needed to run the code.

### Installation Instructions:

1. Create the conda environment with `conda env create -f elm_env.yml`
2. Once installed, activate the environment with `conda activate elm_env`
