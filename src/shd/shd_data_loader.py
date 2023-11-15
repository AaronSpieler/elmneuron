import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_training_batch(example_batch):
    example_batch_input = example_batch[0]
    example_batch_label = example_batch[1]

    fig, axs = plt.subplots(4, 2, figsize=(12, 12))
    axs = axs.flatten()
    for i in range(8):
        x = example_batch_input[i].T
        y = example_batch_label[i]
        axs[i].imshow(
            x,
            cmap=plt.cm.gray_r,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
        )
        axs[i].set_title("Label: " + str(y.item()))
        axs[i].set_ylabel("Input Channel")
        axs[i].set_xlabel("Time Bin")
    fig.tight_layout()
    plt.show()


def random_val_split_SHD_data(train_file, valid_fraction: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    valid_condition = np.array(
        rng.binomial(1, valid_fraction, size=len(np.array(train_file["labels"]))),
        dtype=bool,
    )

    x_train = {
        "times": train_file["spikes"]["times"][~valid_condition],
        "units": train_file["spikes"]["units"][~valid_condition],
    }
    y_train = train_file["labels"][~valid_condition]

    x_valid = {
        "times": train_file["spikes"]["times"][valid_condition],
        "units": train_file["spikes"]["units"][valid_condition],
    }
    y_valid = train_file["labels"][valid_condition]

    return x_train, y_train, x_valid, y_valid


"""
The following code is based on the code by Friedemann Zenke and Manu Halvagal
from https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial4.ipynb
which is licensed under a Creative Commons Attribution
4.0 International License (http://creativecommons.org/licenses/by/4.0/).
The SHD-Adding dataset is a new additional dataset that we propose in our paper.
"""


class SHD(torch.utils.data.IterableDataset):
    def __init__(
        self,
        X: h5py._hl.group.Group,
        y: h5py._hl.dataset.Dataset,
        batch_size: int,
        bin_size: int,
        shuffle: bool = False,
        test_set: bool = True,
        seed: int = 0,
    ):
        super().__init__()

        self.num_input_channel = 700
        self.num_classes = 20
        self.batch_size = batch_size
        self.bin_size = bin_size
        self.shuffle = shuffle
        self.test_set = test_set
        self.seed = seed

        # Load spiking data
        self.labels = np.array(y).astype(np.int32)
        self.firing_times = X["times"]
        self.units_fired = X["units"]

        # Check for test set
        if self.test_set:
            assert (
                len(self.labels) % batch_size == 0
            ), "The dataset size is not exactly divisible by batch size."

        # Epoch management
        self.rng = np.random.default_rng(seed)
        self.sample_index = np.arange(len(self.labels))
        self.batches_per_epoch = len(self.labels) // self.batch_size

        # Precompute time bins
        max_time, in_miliseconds = 1.0, 1000
        step_size = self.bin_size / in_miliseconds
        self.time_bins = np.arange(0, max_time + step_size, step_size)
        self.num_time_bins = len(self.time_bins)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.sample_index)

        for counter in range(len(self)):
            batch_index = self.sample_index[
                self.batch_size * counter : self.batch_size * (counter + 1)
            ]

            coo = [[] for _ in range(3)]
            for batch_count, idx in enumerate(batch_index):
                times = np.digitize(self.firing_times[idx], self.time_bins) - 1
                units = self.units_fired[idx]
                batch = [batch_count for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            # Create sparse matrix
            i = torch.LongTensor(coo)
            v = torch.FloatTensor(np.ones(len(coo[0])))
            X_batch = torch.sparse.FloatTensor(
                i,
                v,
                torch.Size(
                    [self.batch_size, self.num_time_bins, self.num_input_channel]
                ),
            ).to_dense()

            X_batch = X_batch[:, :-1]  # correct for spikes outside interval
            y_batch = torch.Tensor(self.labels[batch_index]).to(torch.int64)

            yield X_batch, y_batch


class SHDAdding(torch.utils.data.IterableDataset):
    """Our proposed SHD-Adding dataset, which consists of concatenated digits
    from the SHD dataset. The target is the sum of the digits.
    """

    def __init__(
        self,
        X: h5py._hl.group.Group,
        y: h5py._hl.dataset.Dataset,
        batch_size: int,
        bin_size: int,
        batches_per_epoch: int = 500,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()

        self.num_input_channel = 700
        self.num_classes = 19
        self.batch_size = batch_size
        self.bin_size = bin_size
        self.shuffle = shuffle
        self.seed = seed

        # Load spiking data
        self.labels = np.array(y).astype(np.int32)
        self.firing_times = X["times"]
        self.units_fired = X["units"]

        # Epoch management
        self.rng = np.random.default_rng(self.seed)
        self.batches_per_epoch = batches_per_epoch

        # Precompute time bins
        max_time, in_miliseconds = 1.0, 1000
        step_size = self.bin_size / in_miliseconds
        self.time_bins = np.arange(0, max_time + step_size, step_size)
        self.num_time_bins = len(self.time_bins)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        if not self.shuffle:
            self.rng = np.random.default_rng(self.seed)

        for i in range(self.batches_per_epoch):
            batch_index = self.rng.integers(
                low=0, high=len(self.labels), size=2 * self.batch_size
            )

            coo = [[] for _ in range(3)]
            for batch_count, idx in enumerate(batch_index):
                times = np.digitize(self.firing_times[idx], self.time_bins) - 1
                units = self.units_fired[idx]
                batch = [batch_count for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo)
            v = torch.FloatTensor(np.ones(len(coo[0])))
            X_batch = torch.sparse.FloatTensor(
                i,
                v,
                torch.Size(
                    [
                        2 * self.batch_size,
                        self.num_time_bins,
                        self.num_input_channel,
                    ]
                ),
            ).to_dense()

            y_batch = self.labels[batch_index]

            arrays_to_concatenate = [
                X_batch[
                    : self.batch_size, :-1, :
                ],  # correct for spikes outside interval
                X_batch[
                    self.batch_size :, :-1, :
                ],  # correct for spikes outside interval
            ]

            X_batch = torch.Tensor(np.concatenate(arrays_to_concatenate, axis=-2))

            # sum two numbers each (irrespective of english or german)
            # labels are 0-9 english, then 0-9 german
            y_batch = torch.Tensor(
                y_batch[: self.batch_size] % 10 + y_batch[self.batch_size :] % 10
            ).to(torch.int64)

            yield X_batch, y_batch
