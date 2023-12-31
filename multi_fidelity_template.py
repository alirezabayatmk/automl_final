"""
===========================
Optimization using BOHB
===========================
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping, Optional
from functools import partial
import time

import numpy as np
import torch

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
    InCondition,
    Categorical
)
from ConfigSpace.read_and_write import json as cs_json
from ConfigSpace.read_and_write import pcs_new, pcs

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac.intensifier.hyperband import Hyperband
from smac.facade import AbstractFacade
from smac.scenario import Scenario
from torch.utils.data import DataLoader, Subset
from dask.distributed import get_worker

from cnn import Model

from datasets import load_deep_woods, load_fashion_mnist

import optuna

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
                "n_channels_conv_0": Integer("n_channels_conv_0", (32, 256), default=256, log=True),
                "n_channels_conv_1": Integer("n_channels_conv_1", (16, 256), default=256, log=True),
                "n_channels_conv_2": Integer("n_channels_conv_2", (16, 256), default=256, log=True),
                "n_fc_layers": Integer("n_fc_layers", (1, 3), default=3),
                "n_channels_fc_0": Integer("n_channels_fc_0", (32, 256), default=256, log=True),
                "n_channels_fc_1": Integer("n_channels_fc_1", (16, 256), default=256, log=True),
                "n_channels_fc_2": Integer("n_channels_fc_2", (16, 256), default=256, log=True),
                "batch_size": Integer("batch_size", (1, 500), default=200, log=True),
                "learning_rate_init": Float(
                    "learning_rate_init",
                    (1e-5, 1e-1),
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
        cs.add_conditions([use_conv_layer_1, use_fc_layer_1])
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
    download = False

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
            logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{5}]")
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


def optimize_hyperparameters(trial):
    # Optimize parameters of the random forest model
    n_trees = trial.suggest_int("n_trees", 10, 25)
    ratio_features = trial.suggest_float("ratio_features", 0.7, 1.0)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    return {
        "n_trees": n_trees,
        "ratio_features": ratio_features,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }


def plot_trajectory(facades: list[AbstractFacade]) -> None:
    """Plots the trajectory (incumbents) of the optimization process."""
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")
    plt.ylabel(facades[0].scenario.objectives)

    for facade in facades:
        X, Y = [], []
        for item in facade.intensifier.trajectory:
            # Single-objective optimization
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            y = item.costs[0]
            x = item.walltime

            X.append(x)
            Y.append(y)

        plt.plot(X, Y, label=facade.intensifier.__class__.__name__)
        plt.scatter(X, Y, marker="x")

    plt.legend()
    plt.show()

    # save the plot
    plt.savefig('trajectory.png')

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
        default=3600,
        type=int,
        help="Running time (seconds) allocated to run the algorithm",
    )
    # 10 default
    parser.add_argument(
        "--max_budget",
        type=float,
        default=10,
        help="maximal budget (image_size) to use with BOHB",
    )
    parser.add_argument(
        "--min_budget", type=float, default=8, help="Minimum budget (image_size) for BOHB"
    )
    parser.add_argument("--eta", type=int, default=2, help="eta for BOHB")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to run the models"
    )
    # default 4
    parser.add_argument(
        "--workers", type=int, default=10, help="num of workers to use with BOHB"
    )
    #default 500
    parser.add_argument(
        "--n_trials", type=int, default=500, help="Number of iterations to run SMAC for"
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
        cs_file=args.configspace
    )

    facades: list[AbstractFacade] = []

    # Optuna applied for optimizing the hyperparameters of the random forest model of SMAC
    def objective(trial):
        rf_params = optimize_hyperparameters(trial)

        scenario = Scenario(
            name="HPO-on-SMAC",
            configspace=configspace,
            deterministic=True,
            output_directory=args.working_dir,
            seed=args.seed,
            n_trials=100,
            max_budget=args.max_budget,
            min_budget=args.min_budget,
            n_workers=10,
            walltime_limit=180,
        )

        smac = SMAC4MF(
            target_function=cnn_from_cfg,
            scenario=scenario,
            initial_design=SMAC4MF.get_initial_design(scenario=scenario, n_configs=5),
            intensifier=Hyperband(
                scenario=scenario,
                incumbent_selection="highest_budget",
                eta=3,
            ),
            model=SMAC4MF.get_model(scenario, **rf_params),
            overwrite=True,
            logging_level=args.log_level,  
        )

        incumbent = smac.optimize()
        return smac.validate(incumbent)
    

    # create Optuna study object and optimize the objective function
    study = optuna.create_study(direction="minimize", study_name="SMAC_HPO")
    study.optimize(objective, n_trials=20, n_jobs=-1)

    best_params = study.best_params
    best_value = study.best_value
    print("Best Optuna Parameters:", best_params)
    print("Best Optuna Value:", best_value)

    with open('best_optuna_params.txt', 'w+') as f:
        f.write(str(best_params))
        f.close()


    # re-run SMAC with the best parameters found by Optuna
    best_scenario = Scenario(
        name="HPO-on-SMAC(best)",
        configspace=configspace,
        deterministic=True,
        output_directory=args.working_dir,
        seed=args.seed,
        n_trials=args.n_trials,
        max_budget=args.max_budget,
        min_budget=args.min_budget,
        n_workers=args.workers,
        walltime_limit=args.runtime
    )

    best_smac = SMAC4MF(
        target_function=cnn_from_cfg,
        scenario=best_scenario,
        initial_design=SMAC4MF.get_initial_design(scenario=best_scenario, n_configs=5),
        intensifier=Hyperband(
            scenario=best_scenario,
            incumbent_selection="highest_budget",
            eta=args.eta,
        ),
        model=SMAC4MF.get_model(best_scenario, **best_params),
        overwrite=True,
        logging_level=args.log_level,  
    )

    best_incumbent = best_smac.optimize()

    default_cost = best_smac.validate(configspace.get_default_configuration())
    print(f"Default cost ({best_smac.intensifier.__class__.__name__}): {default_cost}")

    incumbent_cost = best_smac.validate(best_incumbent)
    print(f"Incumbent cost ({best_smac.intensifier.__class__.__name__}): {incumbent_cost}")

   
    # plot the trajectory of the final optimization process
    facades.append(best_smac)
    plot_trajectory(facades)