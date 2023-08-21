# Import necessary libraries
import argparse
import logging
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Mapping, Optional
from functools import partial
import time

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
    InCondition,
    Categorical,
)
from ConfigSpace.read_and_write import json as cs_json
from ConfigSpace.read_and_write import pcs_new, pcs

from sklearn.model_selection import StratifiedKFold
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario
from torch.utils.data import DataLoader, Subset
from dask.distributed import get_worker

from cnn import Model

from datasets import load_deep_woods, load_fashion_mnist

logger = logging.getLogger(__name__)

CV_SPLIT_SEED = 42


def configuration_space(
        device: str,
        dataset: str,
        cv_count: int = 3,
        budget_type: str = "img_size",
        datasetpath: str | Path = Path("."),
        cs_file: Optional[str | Path] = None
) -> ConfigurationSpace:
    """Build Configuration Space which defines all parameters and their ranges."""
    if cs_file is None:
        # This serves only as an example of how you can manually define a Configuration Space
        # To illustrate different parameter types;
        # we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace(
            {
                "n_conv_layers": Integer("n_conv_layers", (1, 3), default=3),
                "use_BN": Categorical("use_BN", [True, False], default=True),
                "global_avg_pooling": Categorical("global_avg_pooling", [True, False], default=True),
                "n_channels_conv_0": Integer("n_channels_conv_0", (32, 512), default=512, log=True),
                "n_channels_conv_1": Integer("n_channels_conv_1", (16, 512), default=512, log=True),
                "n_channels_conv_2": Integer("n_channels_conv_2", (16, 512), default=512, log=True),
                "n_fc_layers": Integer("n_fc_layers", (1, 3), default=3),
                "n_channels_fc_0": Integer("n_channels_fc_0", (32, 512), default=512, log=True),
                "n_channels_fc_1": Integer("n_channels_fc_1", (16, 512), default=512, log=True),
                "n_channels_fc_2": Integer("n_channels_fc_2", (16, 512), default=512, log=True),
                "batch_size": Integer("batch_size", (1, 1000), default=200, log=True),
                "learning_rate_init": Float(
                    "learning_rate_init",
                    (1e-5, 1.0),
                    default=1e-3,
                    log=True,
                ),
                "kernel_size": Constant("kernel_size", 3),
                "dropout_rate": Constant("dropout_rate", 0.2),
                "device": Constant("device", device),
                "dataset": Constant("dataset", dataset),
                "datasetpath": Constant("datasetpath", str(datasetpath.absolute())),
            }
        )

        # Add conditions to restrict the hyperparameter space
        use_conv_layer_2 = InCondition(cs["n_channels_conv_2"], cs["n_conv_layers"], [3])
        use_conv_layer_1 = InCondition(cs["n_channels_conv_1"], cs["n_conv_layers"], [2, 3])

        use_fc_layer_2 = InCondition(cs["n_channels_fc_2"], cs["n_fc_layers"], [3])
        use_fc_layer_1 = InCondition(cs["n_channels_fc_1"], cs["n_fc_layers"], [2, 3])

        # Add multiple conditions on hyperparameters at once:
        cs.add_conditions([use_conv_layer_2, use_conv_layer_1, use_fc_layer_2, use_fc_layer_1])
    else:
        with open(cs_file, "r") as fh:
            cs_string = fh.read()
            if cs_file.suffix == ".json":
                cs = cs_json.read(cs_string)
            elif cs_file.suffix in [".pcs", ".pcs_new"]:
                cs = pcs_new.read(pcs_string=cs_string)
        logging.info(f"Loaded configuration space from {cs_file}")

        if "device" not in cs:
            cs.add_hyperparameter(Constant("device", device))
        if "dataset" not in cs:
            cs.add_hyperparameter(Constant("dataset", dataset))
        if "cv_count" not in cs:
            cs.add_hyperparameter(Constant("cv_count", cv_count))
        if "budget_type" not in cs:
            cs.add_hyperparameter(Constant("budget_type", budget_type))
        if "datasetpath" not in cs:
            cs.add_hyperparameter(Constant("datasetpath", str(datasetpath.absolute())))
        logging.debug(f"Configuration space:\n{cs}")

    return cs


def get_optimizer_and_criterion(
        cfg: Mapping[str, Any]
) -> tuple[
    type[torch.optim.AdamW | torch.optim.Adam],
    type[torch.nn.MSELoss | torch.nn.CrossEntropyLoss],
]:
    if cfg["optimizer"] == "AdamW":
        model_optimizer = torch.optim.AdamW
    else:
        model_optimizer = torch.optim.Adam

    if cfg["train_criterion"] == "mse":
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss

    return model_optimizer, train_criterion
    

# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
# This is specific to SMAC
def cnn_from_cfg(
        cfg: Configuration,
        seed: int,
        budget: float,
) -> float:
    """
    Creates an instance of the torch_model and fits the given data on it.
    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    :param cfg: Configuration (basically a dictionary)
        configuration chosen by smac
    :param seed: int or RandomState
        used to initialize the rf's random generator
    :param budget: float
        used to set max iterations for the MLP
    Returns
    -------
    val_accuracy cross validation accuracy
    """
    try:
        worker_id = get_worker().name
    except ValueError:
        worker_id = 0

    # If data already existing on disk, set to False
    download = True

    lr = cfg["learning_rate_init"]
    dataset = cfg["dataset"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    ds_path = cfg["datasetpath"]

    # unchangeable constants that need to be adhered to, the maximum fidelities
    img_size = max(8, int(np.floor(budget)))  # example fidelity to use

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    if "fashion_mnist" in dataset:
        input_shape, train_val, _ = load_fashion_mnist(datadir=Path(ds_path, "FashionMNIST"))
    elif "deepweedsx" in dataset:
        input_shape, train_val, _ = load_deep_woods(
            datadir=Path(ds_path, "deepweedsx"),
            resize=(img_size, img_size),
            balanced="balanced" in dataset,
            download=download,
        )
    else:
        raise NotImplementedError

    # returns the cross-validation accuracy
    # to make CV splits consistent
    cv = StratifiedKFold(n_splits=3, random_state=CV_SPLIT_SEED, shuffle=True)

    score = []
    cv_splits = cv.split(train_val, train_val.targets)
    for cv_index, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        logging.info(f"Worker:{worker_id} ------------ CV {cv_index} -----------")
        train_data = Subset(train_val, list(train_idx))
        val_data = Subset(train_val, list(valid_idx))
        
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=batch_size,
            shuffle=False,
        )

        model = Model(
            config=cfg,
            input_shape=input_shape,
            num_classes=len(train_val.classes),
        )
        model = model.to(model_device)

        # summary(model, input_shape, device=device)

        model_optimizer, train_criterion = get_optimizer_and_criterion(cfg)
        optimizer = model_optimizer(model.parameters(), lr=lr)
        train_criterion = train_criterion().to(device)

        for epoch in range(5):  # 20 epochs
            logging.info(f"Worker:{worker_id} " + "#" * 50)
            logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{20}]")
            train_score, train_loss = model.train_fn(
                optimizer=optimizer,
                criterion=train_criterion,
                loader=train_loader,
                device=model_device
            )
            logging.info(f"Worker:{worker_id} => Train accuracy {train_score:.3f} | loss {train_loss}")

        val_score = model.eval_fn(val_loader, device)
        logging.info(f"Worker:{worker_id} => Val accuracy {val_score:.3f}")
        score.append(val_score)

    val_error = 1 - np.mean(score)  # because minimize

    results = val_error
    return results
    

# New: Define multiple fidelities (budget types) with min and max budgets
BUDGET_TYPES = {
    "img_size": {"min": 8, "max": 32, "values": [8, 16, 32]},
    "num_samples": {"min": 500, "max": 10000, "values": [1000, 2000, 4000, 6000, 10000]},
    "num_epochs": {"min": 5, "max": 20, "values": [5, 10, 15, 20]},
    "runtime": {"min": 1000, "max": 21600, "values": [1000, 2000, 5000, 10000, 21600]},
    "cv_folds": {"min": 3, "max": 10, "values": [3, 5, 7, 10]},
}


def run_bohb_for_fidelity(fidelity_type, budget_values):
    for budget_value in budget_values:
        min_budget = BUDGET_TYPES[fidelity_type]["min"]
        max_budget = BUDGET_TYPES[fidelity_type]["max"]
        print(f"Using {fidelity_type} budget: {budget_value}")
        scenario = Scenario(
            name=f"ExampleMFRunWithBOHB_{fidelity_type}_{budget_value}",
            configspace=configspace,
            deterministic=True,
            output_directory=os.path.join(
                args.working_dir, f"results_{fidelity_type}"
            ),
            seed=args.seed,
            n_trials=args.n_trials,
            max_budget=max_budget,
            min_budget=min_budget,
            n_workers=args.workers,
            walltime_limit=min(args.runtime, budget_value),
        )

        smac = SMAC4MF(
            target_function=cnn_from_cfg,
            scenario=scenario,
            initial_design=SMAC4MF.get_initial_design(
                scenario=scenario, n_configs=2
            ),
            intensifier=Hyperband(
                scenario=scenario,
                incumbent_selection="highest_budget",
                eta=args.eta,
            ),
            overwrite=True,
            logging_level=args.log_level,
        )

        incumbent = smac.optimize()


# New: Function to plot results for each fidelity
def plot_results(results, fidelity_type, budget_values):
    plt.figure(figsize=(10, 6))
    plt.plot(budget_values, results, marker="o")
    plt.xlabel(f"Fidelity: {fidelity_type}")
    plt.ylabel("Validation Error")
    plt.title(f"Validation Error vs {fidelity_type}")
    plt.grid(True)
    plt.savefig(
        os.path.join(
            "figures", "fidelities", f"{fidelity_type}_results.png"
        )
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MF example using BOHB.")

    parser.add_argument(
        "--dataset",
        choices=["deepweedsx", "deepweedsx_balanced", "fashion_mnist"],
        default="deepweedsx_balanced",
        help="dataset to use (task for the project: deepweedsx_balanced)",
    )
    parser.add_argument(
        "--working_dir",
        default="./tmp",
        type=str,
        help="directory where intermediate results are stored",
    )
    # 21600 default
    parser.add_argument(
        "--runtime",
        default=1200,
        type=int,
        help="Running time (seconds) allocated to run the algorithm",
    )
    # 10 default
    parser.add_argument(
        "--max_budget",
        type=float,
        default=5,
        help="maximal budget (image_size) to use with BOHB",
    )
    parser.add_argument(
        "--min_budget", type=float, default=1, help="Minimum budget (image_size) for BOHB"
    )
    parser.add_argument("--eta", type=int, default=2, help="eta for BOHB")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to run the models"
    )
    # default 4
    parser.add_argument(
        "--workers", type=int, default=8, help="num of workers to use with BOHB"
    )
    #default 500
    parser.add_argument(
        "--n_trials", type=int, default=100, help="Number of iterations to run SMAC for"
    )
    parser.add_argument(
        "--cv_count",
        type=int,
        default=2,
        help="Number of cross validations splits to create. "
             "Will not have an effect if the budget type is cv_splits",
    )
    parser.add_argument(
        "--log_level",
        choices=[
            "NOTSET"
            "CRITICAL",
            "FATAL",
            "ERROR",
            "WARN",
            "WARNING",
            "INFO",
            "DEBUG",
        ],
        default="NOTSET",
        help="Logging level",
    )
    parser.add_argument('--configspace', type=Path, default="default_configspace.json",
                        help='Path to file containing the configuration space')
    parser.add_argument('--datasetpath', type=Path, default=Path('./data/'),
                        help='Path to directory containing the dataset')

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    configspace = configuration_space(
        device=args.device,
        dataset=args.dataset,
        cv_count=args.cv_count,
        datasetpath=args.datasetpath,
        cs_file=args.configspace,
    )

    # New: Run BOHB optimization for each fidelity type and values
    for fidelity_type, budget_data in BUDGET_TYPES.items():
        budget_values = budget_data["values"]
        run_bohb_for_fidelity(fidelity_type, budget_values)

    # New: Loop to load and plot results for each fidelity type
    for fidelity_type, budget_data in BUDGET_TYPES.items():
        budget_values = budget_data["values"]
        results = []
        for budget_value in budget_values:
            result_file = os.path.join(
                args.working_dir,
                f"results_{fidelity_type}",
                f"SMAC4MF_scenario_results_run_1",
                f"run_1",
                f"traj-run_1",
                f"{budget_value}_SMAC4MF_result.json",
            )
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    result_data = json.load(f)
                results.append(result_data["cost"])
            else:
                results.append(None)
        plot_results(
            results, fidelity_type, budget_values
        )
