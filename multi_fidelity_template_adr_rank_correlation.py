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
from scipy.stats import spearmanr

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

    print('running configuration_space function')

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


    print('configuration_space function finished: ', cs)
    return cs


def get_optimizer_and_criterion(
        cfg: Mapping[str, Any]
) -> tuple[
    type[torch.optim.AdamW | torch.optim.Adam],
    type[torch.nn.MSELoss | torch.nn.CrossEntropyLoss],
]:
    
    print('running get_optimizer_and_criterion function')

    if cfg["optimizer"] == "AdamW":
        model_optimizer = torch.optim.AdamW
    else:
        model_optimizer = torch.optim.Adam

    if cfg["train_criterion"] == "mse":
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss

    print('get_optimizer_and_criterion function finished: ', model_optimizer, train_criterion)

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

    print('running cnn_from_cfg function')

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

    print('cnn_from_cfg function finished: ', lr, dataset, device, batch_size, ds_path)

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
        print('now in cross validation loop')
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

        for epoch in range(20):  # 20 epochs
            print('epoch: ', epoch)
            logging.info(f"Worker:{worker_id} " + "#" * 50)
            logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{20}]")
            train_score, train_loss = model.train_fn(
                optimizer=optimizer,
                criterion=train_criterion,
                loader=train_loader,
                device=model_device
            )
            logging.info(f"Worker:{worker_id} => Train accuracy {train_score:.3f} | loss {train_loss}")
            print('train_score: ', train_score)
            print('train_loss: ', train_loss)

        val_score = model.eval_fn(val_loader, device)
        logging.info(f"Worker:{worker_id} => Val accuracy {val_score:.3f}")
        print('val_score: ', val_score)
        score.append(val_score)

    val_error = 1 - np.mean(score)  # because minimize
    print('val_error: ', val_error)

    results = val_error
    return results


def cnn_from_cfg(
        cfg: Configuration,
        seed: int,
        fidelity: str,
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

    # determine fidelity and used budget
    cv_splits = int(np.floor(budget)) if fidelity == "cv_splits" else 3
    img_size = int(np.floor(budget)) if fidelity == "img_size" else 32
    epochs = int(np.floor(budget)) if fidelity == "epochs" else 20
    sampling_ratio =  int(np.floor(budget)) if fidelity == "epochs" else 100

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

    if fidelity == "sampling_ratio":
        indices = [np.random.choice(len(train_val), int(len(train_val) * sampling_ratio), replace=False)]
        train_val = Subset(train_val, indices)

    # returns the cross-validation accuracy
    # to make CV splits consistent
    cv = StratifiedKFold(n_splits=cv_splits, random_state=CV_SPLIT_SEED, shuffle=True)

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

        for epoch in range(epochs):  # 20 epochs
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

def calc_perf_improv_rates(confs: list[Configuration], fid_budgets: dict) -> dict:
    perf_improv_rates = dict()
    for fidelity, budgets in fid_budgets.items():
        print(f"fidelity {fidelity}: config val acc with {budgets[0]} budget")
        start = time.process_time()
        eval_cheap_perf = [(1 - MLP.train_general(mlp, (fidelity, budgets[0]), conf)) * 100  for conf in confs]
        end = time.process_time()
        print(eval_cheap_perf)
        time_cost_cheap = end - start
        print("Time cost cheap (in s): ", time_cost_cheap)
        print(f"fidelity {fidelity}: config val acc with {budgets[1]} budget")
        start = time.process_time()
        eval_exp_perf = [(1 - MLP.train_general(mlp, (fidelity, budgets[1]), conf)) * 100 for conf in confs]
        end = time.process_time()
        print(eval_exp_perf)
        time_cost_exp = end - start
        print("Time cost expensive (in s): ", time_cost_exp)
        perf_improv = np.mean([np.array(eval_exp_perf) - np.array(eval_cheap_perf)])
        cost_diff = time_cost_exp - time_cost_cheap
        print("Perf. improvement: ", perf_improv)
        print("Cost diff.: ", cost_diff)
        pfr = perf_improv / cost_diff
        print("Performance Improvement Rate: ", pfr)
        perf_improv_rates.update({fidelity: pfr})
    return perf_improv_rates

def calc_spearman_correlation(confs: list[Configuration], seed: int, fidelity_budgets: dict) -> float:
    sp_correlations = dict()

    for fidelity, budgets in fidelity_budgets.items():
        print(f"fidelity: {fidelity}: eval confs with {budgets[0]} budget")
        eval_cheaps = [cnn_from_cfg(conf, seed, fidelity, budgets[0]) for conf in confs]
        print(f"fidelity: {fidelity}: eval confs with {budgets[1]} budget ")
        eval_exp = [cnn_from_cfg(conf, seed, fidelity, budgets[1]) for conf in confs]
        sp_correlations.update({fidelity: spearmanr(eval_cheaps, eval_exp)})

    return sp_correlations

if __name__ == "__main__":
    """
    This is just an example of how to implement BOHB as an optimizer!
    Here we do not consider any of the forbidden clauses.
    """

    print('running main function, parsing arguments')

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
        default=500,
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
        "--min_budget", type=float, default=1, help="Minimum budget (image_size) for BOHB"
    )
    parser.add_argument("--eta", type=int, default=2, help="eta for BOHB")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to run the models"
    )
    # default 4
    parser.add_argument(
        "--workers", type=int, default=16, help="num of workers to use with BOHB"
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
        default="INFO",
        help="Logging level",
    )
    parser.add_argument('--configspace', type=Path, default="default_configspace_edit.json", # previous default
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

    print('determine the best fidelity based on rank correlation')
    start = time.time()
    sample_configs = configspace.sample_configuration(5)
    fidelity_budgets = {'sampling_ratio': (20, 100)} #{'img_size': (8, 16), 'epochs': (5, 10), 'cv_splits': (2, 3),
    sp_rank_corr = calc_spearman_correlation(sample_configs, args.seed, fidelity_budgets)
    print(sp_rank_corr)
    significant_sp_corr = {k: v for k, v in sp_rank_corr.items()}  # if v.pvalue < 0.05}
    sorted_sp_corr = dict(sorted(significant_sp_corr.items(), key=lambda item: item[1].statistic))
    best_fidelity = list(sorted_sp_corr.items())[-1]  # yields the fidelity with the highest spearman rank correlation
    print(sorted_sp_corr)
    print(f"The best fidelity based on the spearman rank correlation is the fidelity {best_fidelity[0]}"
          f" with a spr of {best_fidelity[1]}")
    end = time.time()
    print("Time for determining the best fidelity", end - start)


    print('running main function, config space: ', configspace)

    # Setting up SMAC to run BOHB
    scenario = Scenario(
        name="ExampleMFRunWithBOHB",
        configspace=configspace,
        deterministic=True,
        output_directory=args.working_dir,
        seed=args.seed,
        n_trials=args.n_trials,
        max_budget=fidelity_budgets[best_fidelity[0]][1],
        min_budget=fidelity_budgets[best_fidelity[0]][0],
        n_workers=args.workers,
        walltime_limit=args.runtime
    )

    print('running main function, scenario created: ', scenario)

    # You can mess with SMACs own hyperparameters here (checkout the documentation at https://automl.github.io/SMAC3)
    smac = SMAC4MF(
        target_function=partial(cnn_from_cfg, fidelity=best_fidelity[0]),
        scenario=scenario,
        initial_design=SMAC4MF.get_initial_design(scenario=scenario, n_configs=2),
        intensifier=Hyperband(
            scenario=scenario,
            incumbent_selection="highest_budget",
            eta=args.eta,
        ),
        overwrite=True,
        logging_level=args.log_level,  # https://automl.github.io/SMAC3/main/advanced_usage/8_logging.html
    )

    print('running main function, smac created: ', smac)

    # Start optimization
    incumbent = smac.optimize()

    print('running main function, incumbent finished: ', incumbent)