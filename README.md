## The ELM: an Efficient and Expressive Phenomenological Neuron Model Can Solve Long-Horizon Tasks

> **Note:** This repository provides a minimal PyTorch implementation of the (Branch-)ELM neuron for reproducing the core results of the paper. For the most up-to-date and best-performing implementation (in JAX), please see [elmnetwork](https://github.com/AaronSpieler/elmnetwork).

This repository features a minimal implementation of the (Branch) Expressive Leaky Memory neuron in PyTorch.
Notebooks to train and evaluate on NeuronIO are provided, as well as pre-trained models of various sizes.

![The ELM neuron](./elm_neuron_sketch.png)

### Installation:

1. Create the conda environment with `conda env create -f elm_env.yml`
2. Once installed, activate the environment with `conda activate elm_env`

### Models:

The __models__ folder contains various sized Branch-ELM neuron models pre-trained on NeuronIO.

|  $d_m$    | 1      | 2      | 3      | 5      | 7      | 10     | 15     | 20     | 25     | 30     | 40     | 50     | 75     | 100    |
|-----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| #params   | 4601   | 4708   | 4823   | 5077   | 5363   | 5852   | 6827   | 8002   | 9377   | 10952  | 14702  | 19252  | 34127  | 54002  |
| AUC       | 0.9437 | 0.9582 | 0.9558 | 0.9757 | 0.9827 | 0.9878 | 0.9915 | 0.9922 | 0.9926 | 0.9929 | 0.9934 | 0.9934 | 0.9938 | 0.9935 |

We also include a best effort trained ELM neuron that achieves 0.9946 AUC with $d_m=100$ (0.9949 when evaluated with 1000 ms windows and a 500 ms burn-in, see __models/best_elm_neuron__).

### Notebooks:
- __train_elm_on_shd.ipynb__: train an ELM neuron on SHD or SHD-Adding Dataset.
- __train_elm_on_neuronio.ipynb__: train an ELM neuron on NeuronIO Dataset.
- __eval_elm_on_neuronio.ipynb__: evaluate provided models on the NeuronIO Dataset.
- __neuronio_train_script__: script to train an ELM neuron on NeuronIO Dataset.

### Code:

The __src__ folder contains the implementation and training/evaluation utilities.

- __expressive_leaky_memory_neuron.py__: the implementation of the ELM model.
- __expressive_leaky_memory_neuron_v2.py__: the ELM model with improved timescale handling (see paper), recommended for new projects.
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
- For the input preprocessing, please refer to our paper.

### Known issues:

Specific to this repository (not the code used for the paper) and fixed in [elmnetwork](https://github.com/AaronSpieler/elmnetwork):

- **Best-model selection:** `model.state_dict().copy()` is a shallow copy, so the saved "best" model, including the provided pre-trained models, is always the final-epoch model.
- **Burn-in:** the NeuronIO training loss includes the 150 ms burn-in period excluded in the paper.
- **Evaluation:** `compute_test_predictions` does not pass `y_train_soma_scale` to the model (no effect with the default scale).

Also affecting the paper (to be improved):

- **SHD-Adding:** the test set depends on batch size; reproducing the paper's test set requires the same seed, batch size (8) and number of test batches.

### Corrections to the paper:

The paper misstates two implementation details; the code used for the paper (and this repository) always did the following:

- **Initialization:** Table S1 states Kaiming uniform initialization for the MLP and output weights; the framework default, U(−1/√fan_in, 1/√fan_in), was used.
- **Memory update:** Eq. 1 applies `tanh` to the MLP output; the scaled tanh `1.7159 · tanh(2x/3)` was used (`custom_tanh` in `src/modeling_utils.py`).

### Citation:

If you like what you find, and use an ELM variant or the SHD-Adding dataset, please consider citing us:

```bibtex
@inproceedings{
spieler2024the,
title={The Expressive Leaky Memory Neuron: an Efficient and Expressive Phenomenological Neuron Model Can Solve Long-Horizon Tasks.},
author={Aaron Spieler and Nasim Rahaman and Georg Martius and Bernhard Sch{\"o}lkopf and Anna Levina},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=vE1e1mLJ0U}
}
```
