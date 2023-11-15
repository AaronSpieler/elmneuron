import functools
import glob
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# some fixes for python 3
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import pickle

    basestring = str

# Caching disabled for now
_CACHE_SIZE = 0

# Keep these constants consistent across files
DEFAULT_Y_TRAIN_SOMA_BIAS = -67.7
DEFAULT_Y_SOMA_THRESHOLD = -55.0
DEFAULT_Y_TRAIN_SOMA_SCALE = 1 / 10

# Keep this info as constant available
NEURONIO_SIM_PER_FILE = 128
NEURONIO_SIM_LEN = 6000
NEURONIO_DATA_DIM = 1278
NEURONIO_LABEL_DIM = 2


def determine_python_object_megabyte_size(obj):
    byte_size_estimate = len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    mega_byte_size_estimate = byte_size_estimate / (1024 * 1024)

    return mega_byte_size_estimate


def get_data_files_from_folder(data_folders: list):
    # NOTE: this function only works for the neuronio data set
    train_files = []
    for folder in data_folders:
        train_files.extend(glob.glob(str(Path(folder)) + "/" + "*sim__*"))

    return train_files


def create_neuronio_input_type(num_input=NEURONIO_DATA_DIM):
    # NOTE: this function only works for the neuronio data set
    half_num_input_data = num_input // 2
    input_type = np.concatenate(
        (np.ones(half_num_input_data) * 1, np.ones(half_num_input_data) * -1)
    ).astype(np.int32)

    return input_type


def visualize_training_batch(
    input_spikes,
    target_spikes,
    target_soma,
    pred_spikes=None,
    pred_soma=None,
    num_viz: int = None,
    save_fig_path: str = None,
    y_train_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
):
    input_spikes = input_spikes.cpu().numpy()
    target_spikes = target_spikes.cpu().numpy()
    target_soma = target_soma.cpu().numpy()
    if pred_spikes is not None:
        pred_spikes = pred_spikes.cpu().numpy()
    if pred_soma is not None:
        pred_soma = pred_soma.cpu().numpy()

    # apply train scale to predicitons
    if pred_soma is not None:
        pred_soma *= y_train_scale

    # general visualization settings
    color_palette = sns.color_palette("colorblind")
    sns.set_palette(color_palette)
    sns.set_style("white")

    # spike visualization settings
    levels = [-1.5, -0.5, 0.5, 1.5]
    colors = [color_palette[0], "white", color_palette[3]]
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)

    # configure plot size
    num_viz = target_soma.shape[0] if num_viz is None else num_viz
    sequence_len = target_soma.shape[1]
    rows = num_viz // 2 + num_viz % 2
    columns = 2 if num_viz >= 2 else 1
    fig, axs = plt.subplots(
        nrows=rows, ncols=columns, figsize=(9 if num_viz >= 2 else 7.5, 3 * rows)
    )
    ax = axs.flatten() if num_viz >= 2 else [axs]

    for i in range(num_viz):
        lines = []
        legends = []

        # configure general grid
        ax[i].yaxis.set_visible(False)
        ax[i].set_xlabel("Time (ms)")
        ax[i].xaxis.grid(False)

        # plot input spikes
        ax1 = ax[i].twinx()
        ax1.imshow(
            input_spikes[i].T,
            cmap=cmap,
            origin="lower",
            norm=norm,
            interpolation="nearest",
            aspect="auto",
        )
        ax1.yaxis.set_visible(False)

        # configure voltage axis
        ax2 = ax[i].twinx()
        ax2.set_ylim(-2.5, 5.5)
        ax2.set_xlim(0, sequence_len)
        ax2.set_ylabel(f"Voltage ({1/y_train_scale} mV)")
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position("left")

        # plot target voltage
        (line,) = ax2.plot(
            target_soma[i],
            label="Target Voltage",
            color=color_palette[1],
            linewidth=1.0,
        )
        lines.append(line)

        if pred_soma is not None:
            # plot pred soma
            (line,) = ax2.plot(
                pred_soma[i],
                label="Pred. Voltage",
                color=color_palette[0],
                linewidth=1.0,
            )
            lines.append(line)

        # plot target spikes
        nonzero_target_spikes = np.nonzero(target_spikes[i])
        if len(nonzero_target_spikes) > 0:
            line = ax2.vlines(
                x=nonzero_target_spikes,
                ymin=1.5,
                ymax=2.1,
                colors=color_palette[4],
                label=f"Target Spikes (N={int(sum(target_spikes[i]))})",
                linewidth=1.0,
            )
            lines.append(line)

        if pred_spikes is not None:
            # configure spikre prediction axis
            ax3 = ax[i].twinx()
            ax3.set_ylim(-0.1, 1.1)
            ax3.set_ylabel(f"Spike Probability")
            ax3.grid(False)

            # plot pred spikes
            (line,) = ax3.plot(
                pred_spikes[i],
                label="Pred. Spike Probabilty",
                color=color_palette[2],
                linewidth=1.0,
            )
            lines.append(line)

        # combine legends
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels)

        # set the border color to grey
        for spine in ax[i].spines.values():
            spine.set_edgecolor("grey")
        for spine in ax2.spines.values():
            spine.set_edgecolor("grey")

        # enable x axis spines
        ax[i].xaxis.set_ticks_position("bottom")

    fig.tight_layout()
    if save_fig_path is not None:
        fig.savefig(save_fig_path, dpi=300)
    else:
        plt.show()


"""
Most of the following code was written by David Beniaguev and Oren Amsalem and originates
from https://github.com/SelfishGene/neuron_as_deep_net/blob/master/fit_CNN.py.
Main changes include adding the include_params, encoding and verbose options
and related code insertions and modificaitons.
"""


def bin2dict(bin_spikes_matrix):
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds, spike_times):
        if row_ind in row_inds_spike_times_map.keys():
            row_inds_spike_times_map[row_ind].append(syn_time)
        else:
            row_inds_spike_times_map[row_ind] = [syn_time]

    return row_inds_spike_times_map


def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms):
    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype="bool")
    for row_ind in row_inds_spike_times_map.keys():
        for spike_time in row_inds_spike_times_map[row_ind]:
            bin_spikes_matrix[row_ind, spike_time] = 1.0

    return bin_spikes_matrix


@functools.lru_cache(maxsize=_CACHE_SIZE, typed=True)
def parse_sim_experiment_file(
    sim_experiment_file: str, include_params: bool = False, verbose=False, encoding=None
):
    if verbose:
        print("-----------------------------------------------------------------")
        print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")
    loading_start_time = time.time()

    with open(sim_experiment_file, "rb") as file:
        if encoding is not None:
            file.seek(0)
            experiment_dict = pickle.load(file, encoding=encoding)
        else:
            try:
                file.seek(0)
                experiment_dict = pickle.load(file, encoding="ASCII")
            except UnicodeDecodeError:
                file.seek(0)
                experiment_dict = pickle.load(file, encoding="latin1")

    # gather params
    num_simulations = len(experiment_dict["Results"]["listOfSingleSimulationDicts"])
    num_segments = len(experiment_dict["Params"]["allSegmentsType"])
    sim_duration_ms = experiment_dict["Params"]["totalSimDurationInSec"] * 1000
    num_ex_synapses = num_segments
    num_inh_synapses = num_segments
    num_synapses = num_ex_synapses + num_inh_synapses

    # returning some data properties
    if include_params:
        params = dict()
        params["num_simulations"] = num_simulations
        params["num_segments"] = num_segments
        params["sim_duration_ms"] = sim_duration_ms
        params["num_ex_synapses"] = num_ex_synapses
        params["num_inh_synapses"] = num_inh_synapses
        params["num_synapses"] = num_synapses
        params["synapse_type"] = create_neuronio_input_type().tolist()
        params["segment_type"] = experiment_dict["Params"]["allSegmentsType"]
        params["segment_dist_from_soma"] = experiment_dict["Params"][
            "allSegments_DistFromSoma"
        ]

        # copied values from section "Show filter of a selected unit in depth." of:
        # https://www.kaggle.com/code/selfishgene/single-neuron-as-deep-net-replicating-key-result
        if num_segments == 639:
            basal_cutoff = 262
            tuft_cutoff = [366, 559]

            ex_basal_syn_inds = np.arange(basal_cutoff)
            ex_oblique_syn_inds = np.hstack(
                (
                    np.arange(basal_cutoff, tuft_cutoff[0]),
                    np.arange(tuft_cutoff[1], num_segments),
                )
            )
            ex_tuft_syn_inds = np.arange(tuft_cutoff[0], tuft_cutoff[1])

            params["ex_basal_syn_inds"] = ex_basal_syn_inds.tolist()
            params["ex_oblique_syn_inds"] = ex_oblique_syn_inds.tolist()
            params["ex_tuft_syn_inds"] = ex_tuft_syn_inds.tolist()
            params["inh_basal_syn_inds"] = (num_segments + ex_basal_syn_inds).tolist()
            params["inh_oblique_syn_inds"] = (
                num_segments + ex_oblique_syn_inds
            ).tolist()
            params["inh_tuft_syn_inds"] = (num_segments + ex_tuft_syn_inds).tolist()

    # collect X, y_spike, y_soma
    X = np.zeros((num_synapses, sim_duration_ms, num_simulations), dtype="bool")
    y_spike = np.zeros((sim_duration_ms, num_simulations))
    y_soma = np.zeros((sim_duration_ms, num_simulations))
    for k, sim_dict in enumerate(
        experiment_dict["Results"]["listOfSingleSimulationDicts"]
    ):
        X_ex = dict2bin(sim_dict["exInputSpikeTimes"], num_segments, sim_duration_ms)
        X_inh = dict2bin(sim_dict["inhInputSpikeTimes"], num_segments, sim_duration_ms)
        X[:, :, k] = np.vstack((X_ex, X_inh))
        spike_times = (sim_dict["outputSpikeTimes"].astype(float) - 0.5).astype(int)
        y_spike[spike_times, k] = 1.0
        y_soma[:, k] = sim_dict["somaVoltageLowRes"]

    loading_duration_sec = time.time() - loading_start_time
    if verbose:
        print("loading took %.3f seconds" % (loading_duration_sec))
        print("-----------------------------------------------------------------")

    if include_params:
        return X, y_spike, y_soma, params

    return X, y_spike, y_soma
