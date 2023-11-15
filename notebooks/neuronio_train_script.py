# Imports
import gc
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Relative Imports
package_path = Path(...)  # TODO: change to elmneuron path
sys.path.insert(0, str(package_path))

from src.expressive_leaky_memory_neuron import ELM
from src.neuronio.neuronio_data_loader import NeuronIO
from src.neuronio.neuronio_data_utils import (
    NEURONIO_DATA_DIM,
    NEURONIO_LABEL_DIM,
    get_data_files_from_folder,
    visualize_training_batch,
)
from src.neuronio.neuronio_eval_utils import (
    NeuronioEvaluator,
    compute_test_predictions_multiple_sim_files,
    filter_and_extract_core_results,
    get_num_trainable_params,
)
from src.neuronio.neuronio_train_utils import NeuronioLoss

if __name__ == "__main__":
    ########## Logging Config ##########
    print("Wandb configuration started...")

    # setup directory for saving training artefacts
    temporary_dir = tempfile.TemporaryDirectory()
    artefacts_dir = Path(temporary_dir.name) / "training_artefacts"
    os.makedirs(str(artefacts_dir))

    # wandb config
    api_key_file = ...  # TODO: change to api_key path
    project_name = ...  # TODO: change to project name
    group_name = ...  # TODO: change to group name

    # login to wandb
    with open(api_key_file, "r") as file:
        api_key = file.read().strip()
    wandb.login(key=api_key)

    # initialize wandb
    wandb.init(project=project_name, group=group_name, config={})

    ########## General Config ##########
    print("General configuration started...")

    # General Config
    general_config = dict()
    general_config["seed"] = 0
    general_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    general_config["short_training_run"] = False
    general_config["verbose"] = general_config["short_training_run"]
    torch_device = torch.device(general_config["device"])
    print("Torch Device: ", torch_device)

    # Seeding & Determinism
    os.environ["PYTHONHASHSEED"] = str(general_config["seed"])
    random.seed(general_config["seed"])
    np.random.seed(general_config["seed"])
    torch.manual_seed(general_config["seed"])
    torch.cuda.manual_seed(general_config["seed"])
    torch.backends.cudnn.deterministic = True

    ########## Data, Model and Training Config ##########
    print("Data, model and training configuration started...")

    # NOTE: this step requires you having downloaded the dataset

    # Download Train Data:
    # https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-train-data
    # Download Test Data:
    # https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-test-data # Data_test

    # Location of downloaded folders
    data_dir_path = Path(...)  # TODO: change to neuronio data path
    train_data_dir_path = (
        data_dir_path / "neuronio_train_data"
    )  # TODO: change to train subfolder
    test_data_dir_path = (
        data_dir_path / "neuronio_test_data"
    )  # TODO: change to test subfolder

    # Data Config

    data_config = dict()
    train_data_dirs = [
        str(train_data_dir_path / "full_ergodic_train_batch_2"),
        str(train_data_dir_path / "full_ergodic_train_batch_3"),
        str(train_data_dir_path / "full_ergodic_train_batch_4"),
        str(train_data_dir_path / "full_ergodic_train_batch_5"),
        str(train_data_dir_path / "full_ergodic_train_batch_6"),
        str(train_data_dir_path / "full_ergodic_train_batch_7"),
        str(train_data_dir_path / "full_ergodic_train_batch_8"),
        str(train_data_dir_path / "full_ergodic_train_batch_9"),
        str(train_data_dir_path / "full_ergodic_train_batch_10"),
    ]
    valid_data_dirs = [str(train_data_dir_path / "full_ergodic_train_batch_1")]
    test_data_dirs = [str(test_data_dir_path)]

    data_config["train_data_dirs"] = train_data_dirs
    data_config["valid_data_dirs"] = valid_data_dirs
    data_config["test_data_dirs"] = test_data_dirs

    data_config["data_dim"] = NEURONIO_DATA_DIM
    data_config["label_dim"] = NEURONIO_LABEL_DIM

    # Model Config

    model_config = dict()
    model_config["num_input"] = data_config["data_dim"]
    model_config["num_output"] = data_config["label_dim"]
    model_config["num_memory"] = 20  # TODO: change to desired value
    model_config["memory_tau_min"] = 1.0
    model_config["memory_tau_max"] = 150.0
    model_config["learn_memory_tau"] = False
    model_config["num_branch"] = 45
    model_config["num_synapse_per_branch"] = 100
    model_config["input_to_synapse_routing"] = "neuronio_routing"

    # Training Config

    train_config = dict()
    train_config["num_epochs"] = 5 if general_config["short_training_run"] else 35
    train_config["learning_rate"] = 5e-4
    train_config["batch_size"] = 32 if general_config["short_training_run"] else 8
    train_config["batches_per_epoch"] = (
        1000 if general_config["short_training_run"] else 10000
    )
    train_config["batches_per_epoch"] = int(
        8 / train_config["batch_size"] * train_config["batches_per_epoch"]
    )
    train_config["file_load_fraction"] = (
        0.5 if general_config["short_training_run"] else 0.3
    )
    train_config["num_prefetch_batch"] = 100
    train_config["num_workers"] = 10
    train_config["burn_in_time"] = 150
    train_config["input_window_size"] = 500

    # Save Configs

    with open(str(artefacts_dir / "general_config.json"), "w", encoding="utf-8") as f:
        json.dump(general_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artefacts_dir / "data_config.json"), "w", encoding="utf-8") as f:
        json.dump(data_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artefacts_dir / "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    with open(str(artefacts_dir / "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=4, sort_keys=True)
    wandb.config.update(
        {
            "general_config": general_config,
            "data_config": data_config,
            "model_config": model_config,
            "train_config": train_config,
        }
    )

    ########## Data, Model and Training Setup ##########
    print("Data, model and training setup started...")

    # Preparing Data Loaders

    train_files = get_data_files_from_folder(data_config["train_data_dirs"])
    valid_files = get_data_files_from_folder(data_config["valid_data_dirs"])
    test_files = get_data_files_from_folder(data_config["test_data_dirs"])

    train_data_loader = NeuronIO(
        file_paths=train_files,
        batches_per_epoch=train_config["batches_per_epoch"],
        batch_size=train_config["batch_size"],
        input_window_size=train_config["input_window_size"],
        num_workers=train_config["num_workers"],
        num_prefetch_batch=train_config["num_prefetch_batch"],
        file_load_fraction=train_config["file_load_fraction"],
        seed=general_config["seed"],
        device=torch_device,
    )

    train_evaluator = NeuronioEvaluator(
        test_file=train_files[0],
        burn_in_time=train_config["burn_in_time"],
        input_window_size=train_config["input_window_size"],
        device=torch_device,
    )

    valid_evaluator = NeuronioEvaluator(
        test_file=valid_files[0],
        burn_in_time=train_config["burn_in_time"],
        input_window_size=train_config["input_window_size"],
        device=torch_device,
    )

    # Visualize training data
    X_viz, (y_spike_viz, y_soma_viz) = next(iter(train_data_loader))
    visualize_training_batch(
        X_viz,
        y_spike_viz,
        y_soma_viz,
        num_viz=8,
        save_fig_path=str(artefacts_dir / "training_batch.png"),
    )

    # Initialize the ELM model
    model = ELM(**model_config).to(torch_device)

    # Initialize the loss function, optimizer, and scheduler
    criterion = NeuronioLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    scheduler = CosineAnnealingLR(
        optimizer, T_max=train_config["batches_per_epoch"] * train_config["num_epochs"]
    )

    # Visualize ELM model
    print(model)

    ########## TEAINING ##########
    print("Training started...")

    # Initialize the best validation RMSE to a high value
    best_valid_rmse = float("inf")
    best_model_state_dict = model.state_dict().copy()

    # Training loop
    train_rmse_hist = []
    train_auc_hist = []
    valid_rmse_hist = []
    valid_auc_hist = []
    for epoch in range(train_config["num_epochs"]):
        # Training
        model.train()
        running_loss = 0.0
        pbar = tqdm(
            enumerate(train_data_loader, 0),
            total=train_config["batches_per_epoch"],
            disable=not general_config["verbose"],
        )
        for i, data in pbar:
            inputs, targets = data

            # Perform a single training step
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update running loss
            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1} Loss: {running_loss / (i+1):.5f}")

        model.eval()
        with torch.no_grad():
            # Evaluate on training data
            train_eval_metrics = train_evaluator.evaluate(model)
            train_rmse = train_eval_metrics["soma_RMSE"]
            train_auc = train_eval_metrics["AUC"]
            train_rmse_hist.append(train_rmse)
            train_auc_hist.append(train_auc)

            # Evaluate on validation data
            valid_eval_metrics = valid_evaluator.evaluate(model)
            valid_rmse = valid_eval_metrics["soma_RMSE"]
            valid_auc = valid_eval_metrics["AUC"]
            valid_rmse_hist.append(valid_rmse)
            valid_auc_hist.append(valid_auc)

        # Copy model state dict if validation RMSE has improved
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_model_state_dict = model.state_dict().copy()

        # Print statistics
        print(
            f"Epoch: {epoch+1}, "
            f'Train Loss: {running_loss / train_config["batches_per_epoch"]:.5f}, '
            f"Train RMSE: {train_rmse:.5f}, Train AUC: {train_auc:.5f}, "
            f"Valid RMSE: {valid_rmse:.5f}, Valid AUC: {valid_auc:.5f}"
        )

        # Log statistics
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / train_config["batches_per_epoch"],
                "train_rmse": train_rmse,
                "train_auc": train_auc,
                "valid_rmse": valid_rmse,
                "valid_auc": valid_auc,
            }
        )

    # Free up memory
    del train_data_loader
    gc.collect()

    ########## EVALUATION ##########
    print("Evaluation started...")

    # Load the best model for evaluation
    model.load_state_dict(best_model_state_dict)
    model.eval()

    # Visualize predictions
    with torch.no_grad():
        outputs = model.neuronio_eval_forward(X_viz)
        visualize_training_batch(
            X_viz,
            y_spike_viz,
            y_soma_viz,
            outputs[..., 0],
            outputs[..., 1],
            num_viz=8,
            save_fig_path=str(artefacts_dir / "training_batch_inference.png"),
        )

    # Gather all evaluation metrics
    with torch.no_grad():
        if not general_config["short_training_run"]:
            random.seed(general_config["seed"])
            select_train_files = random.choices(train_files, k=10)
            train_predictions = compute_test_predictions_multiple_sim_files(
                neuron=model,
                test_files=select_train_files,
                burn_in_time=train_config["burn_in_time"],
                input_window_size=train_config["input_window_size"],
                device=torch_device,
            )
            train_results = filter_and_extract_core_results(
                *train_predictions, verbose=False
            )

            valid_predictions = compute_test_predictions_multiple_sim_files(
                neuron=model,
                test_files=valid_files,
                burn_in_time=train_config["burn_in_time"],
                input_window_size=train_config["input_window_size"],
                device=torch_device,
            )
            valid_results = filter_and_extract_core_results(
                *valid_predictions, verbose=False
            )

        test_predictions = compute_test_predictions_multiple_sim_files(
            neuron=model,
            test_files=test_files,
            burn_in_time=train_config["burn_in_time"],
            input_window_size=train_config["input_window_size"],
            device=torch_device,
        )
        test_results = filter_and_extract_core_results(*test_predictions, verbose=False)

    eval_results = dict()
    if not general_config["short_training_run"]:
        eval_results["train_results"] = train_results
        eval_results["valid_results"] = valid_results
    eval_results["test_results"] = test_results

    # Show evaluation results
    print(eval_results)

    ########## SERIALIZATION ##########
    print("Serialization started...")

    model_stats = dict()
    # calculate model params and flops
    with torch.no_grad():
        model_stats["model_param_count"] = get_num_trainable_params(model)
        model_stats["model_flop_count"] = None  # TODO: use appropirate package

    # save eval results
    with open(str(artefacts_dir / "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4, sort_keys=True)
    wandb.log(eval_results)
    with open(str(artefacts_dir / "model_stats.json"), "w", encoding="utf-8") as f:
        json.dump(model_stats, f, ensure_ascii=False, indent=4, sort_keys=True)
    wandb.log({"model_stats": model_stats})

    # save model for later
    torch.save(
        best_model_state_dict, str(artefacts_dir / "neuronio_best_model_state.pt")
    )

    # TODO: copy artefacts_dir for local version

    # save artefacts to wandb
    wandb.save(
        str(artefacts_dir) + "/*", base_path=str(temporary_dir.name), policy="now"
    )
    wandb.finish()  # finish wandb run
    temporary_dir.cleanup()

    ########## FINISHED ##########
    print("Finished")
