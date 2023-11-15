## The ELM: an Efficient and Expressive Phenomenological Neuron Model Can Solve Long-Horizon Tasks

This repository features a minimal implementation of the (Branch) Expressive Leaky Memory neuron in PyTorch.
Notebooks to train and evaluate on NeuronIO are provided, as well as pre-trained models of various sizes.

### Installation:

1. Create the conda environment with `conda env create -f elm_env.yml`
2. Once installed, activate the environment with `conda activate elm_env`

### Models:

The __models__ folder contains various sized Branch-ELM neuron models pre-trained on NeuronIO.

|  $d_m$    | 1      | 2      | 3      | 5      | 7      | 10     | 15     | 20     | 25     | 30     | 40     | 50     | 75     | 100    |
|-----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| #params   | 4601   | 4708   | 4823   | 5077   | 5363   | 5852   | 6827   | 8002   | 9377   | 10952  | 14702  | 19252  | 34127  | 54002  |
| AUC       | 0.9437 | 0.9582 | 0.9558 | 0.9757 | 0.9827 | 0.9878 | 0.9915 | 0.9922 | 0.9926 | 0.9929 | 0.9934 | 0.9934 | 0.9938 | 0.9935 |

### Notebooks:
- __train_elm_on_shd.ipynb__: train an ELM neuron on SHD or SHD-Adding Dataset.
- __train_elm_on_neuronio.ipynb__: train an ELM neuron on NeuronIO Dataset.
- __eval_elm_on_neuronio.ipynb__: evaluate provided models on the NeuronIO Dataset.
- __neuronio_train_script__: script to train an ELM neuron on NeuronIO Dataset.

### Code:

The __src__ folder contains the implementation and training/evaluation utilities.

- __expressive_leaky_memory_neuron.py__: the implementation of the ELM model.
- __neuronio__: files related to visualising, training and evaluating on the NeuronIO dataset.
- __shd__: files related to downloading, training and evaluating on the Spiking Heidelberg Digits (SHD) dataset, and its SHD-Adding version.

Note: the PyTorch implementation seems to be about 2x slower than the jax version unfortunately.

### Dataset:

Running the NeuronIO related code requires downloading the dataset first (~115GB).

- Download Train Data: [single-neurons-as-deep-nets-nmda-train-data](https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-train-data)
- Download Test Data (Data_test): [single-neurons-as-deep-nets-nmda-test-data](https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-test-data)
- For more information, please checkout the following repository: [neuron_as_deep_net](https://github.com/SelfishGene/neuron_as_deep_net)

Running the SHD related code is possible without seperately downloading the dataset (~0.5GB).

- The small SHD daset will automaticall be downloaded upon running the related notebook.
- A dataloader for the introduced SHD-Adding dataset is provided in __/src/shd/shd_data_loader.py__
- For more information onf SHD, please checkout the following website: [spiking-heidelberg-datasets-shd](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)

Running the LRA training/evaluation is not provided at the moment.

- To download the dataset, we recommend to checkout the following repository: [mega](https://github.com/facebookresearch/mega)
- For the input preprocessing, please refer to our preprint.

### Citation:

If you like what you find, and use an ELM variant or the SHD-Adding dataset, please consider citing us:

[1] Spieler, A., Rahaman, N., Martius, G., Schölkopf, B., & Levina, A. (2023). The ELM Neuron: an Efficient and Expressive Cortical Neuron Model Can Solve Long-Horizon Tasks. arXiv preprint arXiv:2306.16922.
