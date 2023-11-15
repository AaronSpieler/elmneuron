import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ..modeling_utils import create_interlocking_indices
from .neuronio_data_utils import DEFAULT_Y_SOMA_THRESHOLD, DEFAULT_Y_TRAIN_SOMA_BIAS


def visualize_neuron_workings(
    neuron,
    input_spikes,
    target_spikes=None,
    target_soma=None,
    burn_in_time: int = None,
    syn_sample_values: int = 20,
    mem_sample_values: int = None,
    color_by_memory_tau: bool = False,
    y_train_soma_bias: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
    y_soma_threshold: float = DEFAULT_Y_SOMA_THRESHOLD,
    save_fig_path: str = None,
    seed=0,
    device="cpu",
):
    # Visualization settings
    master_scale = 1.8
    color_palette = sns.color_palette("colorblind")
    sns.set_palette(color_palette)
    sns.set(font_scale=master_scale, rc={"lines.linewidth": 0.5 + master_scale})
    sns.set_style("white")

    # Make copy of inputs
    input_spikes = np.copy(input_spikes)
    target_soma = np.copy(target_soma)
    target_spikes = np.copy(target_spikes)

    # Compute model predictions
    input_spikes_torch = torch.from_numpy(input_spikes).to(device)
    input_spikes_torch = torch.unsqueeze(input_spikes_torch, 0)
    outputs, s_records, m_records = neuron.neuronio_viz_forward(input_spikes_torch)

    # Convert inference artifacts
    outputs = outputs.squeeze(0)
    pred_spikes = outputs[..., 0].detach().cpu().numpy()
    pred_voltages = outputs[..., 1].detach().cpu().numpy()
    all_syn_values = s_records.squeeze(0).detach().cpu().T.numpy()
    all_mem_values = m_records.squeeze(0).detach().cpu().T.numpy()

    # Randomly subsample hidden states
    sample_rng = np.random.default_rng(seed)
    if syn_sample_values is not None:
        syn_sample_indices = sample_rng.choice(
            all_syn_values.shape[0], size=syn_sample_values, replace=False
        )
        all_syn_values = all_syn_values[syn_sample_indices]
    if mem_sample_values is not None:
        mem_sample_indices = sample_rng.choice(
            all_mem_values.shape[0], size=mem_sample_values, replace=False
        )
        all_mem_values = all_mem_values[mem_sample_indices]

    # Create subplots
    fig, axs = plt.subplots(4, figsize=(16, 14), sharex=True)

    # --------------------- PLOTS INPUT SPIKES ---------------------

    curr_ax = axs[0]
    curr_ax.annotate("a)", xy=(-0.1, 0.951), xycoords="axes fraction", fontsize=30)
    curr_ax.set_ylabel("Synapse Index")
    curr_ax.set_ylim(0, len(input_spikes[0]))
    input_axis = curr_ax

    # scatter plot input
    interlocking_indices = create_interlocking_indices(input_spikes.shape[1])
    x_y_indices = np.where(np.abs(input_spikes[:, interlocking_indices]) > 0.5)
    colors = np.array(["red" if y % 2 == 0 else "blue" for y in x_y_indices[1]])
    curr_ax.scatter(*x_y_indices, s=0.05, alpha=0.8, color=colors)

    # for legend
    curr_ax.scatter(-100, -100, color="blue", label="Inhibitory Input", alpha=0.8)
    curr_ax.scatter(-100, -100, color="red", label="Excitatory Input", alpha=0.8)

    # --------------------- PLOTS SOMA VOLTAGES ---------------------

    curr_ax = axs[1]
    curr_ax.annotate("b)", xy=(-0.1, 0.951), xycoords="axes fraction", fontsize=30)
    curr_ax.set_ylabel("Membrane\n Voltage (mV)")
    curr_ax.set_ylim(-90, -30)
    curr_ax.set_xlim(0, len(pred_voltages))
    curr_ax.xaxis.grid(False)
    soma_axis = curr_ax

    # plot predictions
    pred_voltages += y_train_soma_bias
    curr_ax.plot(pred_voltages, label="Predicted Voltage", color=color_palette[0])

    # plot ground truth
    if target_soma is not None:
        curr_ax.plot(target_soma, label="Target Voltage", color=color_palette[1])

    # --------------------- PLOTS SPIKE PROBABILITIES ---------------------

    curr_ax = soma_axis.twinx()
    curr_ax.set_ylabel("Spike Probability")
    curr_ax.set_ylim(-0.1, 1.1)
    curr_ax.grid(False)

    # plot predictions
    curr_ax.plot(pred_spikes, color=color_palette[2])

    # plot ground truth
    if target_spikes is not None:
        nonzero_target_spikes = np.nonzero(target_spikes)[0]
        soma_axis.vlines(
            x=nonzero_target_spikes,
            ymin=y_soma_threshold + 5,
            ymax=y_soma_threshold + 20,
            colors=color_palette[4],
            label=f"Target Spikes (N={int(sum(target_spikes))})",
        )

    # for legend
    soma_axis.plot(
        -100, -100, label="Predicted Spike Probability", color=color_palette[2]
    )

    # --------------------- LEGENDS ---------------------

    input_axis.legend(fontsize=18, loc="upper right")
    soma_axis.legend(fontsize=18, loc="upper center")

    # --------------------- PLOTS SYNAPSE VALUES ---------------------

    curr_ax = axs[2]
    curr_ax.annotate("c)", xy=(-0.1, 0.951), xycoords="axes fraction", fontsize=30)
    curr_ax.annotate(
        f"N={len(all_syn_values)}",
        xy=(0.93, 0.85),
        xycoords="axes fraction",
        fontsize=19,
    )
    curr_ax.set_ylabel("Synapse Values")
    max_val = 1.05 * np.max(np.abs(all_syn_values))
    curr_ax.set_ylim(-max_val, max_val)
    synapse_axis = curr_ax

    # plot synapse values
    curr_ax.plot(all_syn_values.T, alpha=0.7)

    # --------------------- PLOTS MEMORY VALUES ---------------------

    # create a colormap
    memor_tau_values = neuron.tau_m
    mem_cmap = cm.get_cmap("viridis")
    mem_norm = mcolors.Normalize(vmin=min(memor_tau_values), vmax=max(memor_tau_values))

    curr_ax = axs[3]
    curr_ax.annotate("d)", xy=(-0.1, 0.951), xycoords="axes fraction", fontsize=30)
    curr_ax.annotate(
        f"N={len(all_mem_values)}",
        xy=(0.93, 0.85),
        xycoords="axes fraction",
        fontsize=19,
    )
    curr_ax.set_ylabel("Memory Values")
    curr_ax.set_xlabel("Time (ms)")
    max_val = 1.05 * np.max(np.abs(all_mem_values))
    curr_ax.set_ylim(-max_val, max_val)

    if color_by_memory_tau:
        # plot memory values by timescale
        for idx, mem_values in enumerate(all_mem_values):
            tau = memor_tau_values[idx]
            curr_ax.plot(mem_values, alpha=0.7, color=mem_cmap(mem_norm(tau)))

        # color bar configuration
        cbar_ax_position = curr_ax.get_position()
        cbar_ax = fig.add_axes([cbar_ax_position.x1 + 0.035, 0.075, 0.015, 0.207])
        cbar_ax.set_title(r"$\tau_m$")
        sm = cm.ScalarMappable(norm=mem_norm, cmap=mem_cmap)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)
        cbar_ax.yaxis.set_label_position("left")

    else:
        # plot memory values
        curr_ax.plot(all_mem_values.T)

    # --------------------- PLOTS BURN_IN PERIOD ---------------------

    # If a burn-in time is specified, fill between the axes
    if burn_in_time is not None:
        for ax in axs:
            ax.fill_between([0, burn_in_time], *ax.get_ylim(), color="grey", alpha=0.3)

    # --------------------- FINALIZE PLOT ---------------------

    # Set the border color to grey and enable y-axis spines
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_edgecolor("grey")
        ax.yaxis.set_ticks_position("left")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    if save_fig_path is not None:
        fig.savefig(save_fig_path, dpi=300)
    else:
        plt.show()
