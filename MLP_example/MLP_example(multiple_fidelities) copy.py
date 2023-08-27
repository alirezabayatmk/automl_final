import math
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from functools import partial
import logging

# Disable error logs for scikit-learn
logging.getLogger("sklearn").setLevel(logging.ERROR)


dataset = load_digits()



def random_subset(dataset, sampling_ratio):
    if sampling_ratio < 0 or sampling_ratio > 100:
        raise ValueError("Sampling ratio should be between 0 and 100.")

    num_samples = int(len(dataset.data) * (sampling_ratio / 100))
    random_indices = np.random.choice(len(dataset.data), num_samples, replace=False)

    subset_data = dataset.data[random_indices]
    subset_target = dataset.target[random_indices]

    return subset_data, subset_target

class MLP:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types, we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace()

        n_layer = Integer("n_layer", (1, 5), default=1)
        n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
        activation = Categorical("activation", ["logistic", "tanh", "relu"], default="tanh")
        solver = Categorical("solver", ["lbfgs", "sgd", "adam"], default="adam")
        batch_size = Integer("batch_size", (30, 300), default=200)
        learning_rate = Categorical("learning_rate", ["constant", "invscaling", "adaptive"], default="constant")
        learning_rate_init = Float("learning_rate_init", (0.0001, 1.0), default=0.001, log=True)

        # Add all hyperparameters at once:
        cs.add_hyperparameters([n_layer, n_neurons, activation, solver, batch_size, learning_rate, learning_rate_init])

        # Adding conditions to restrict the hyperparameter space...
        # ... since learning rate is only used when solver is 'sgd'.
        use_lr = EqualsCondition(child=learning_rate, parent=solver, value="sgd")
        # ... since learning rate initialization will only be accounted for when using 'sgd' or 'adam'.
        use_lr_init = InCondition(child=learning_rate_init, parent=solver, values=["sgd", "adam"])
        # ... since batch size will not be considered when optimizer is 'lbfgs'.
        use_batch_size = InCondition(child=batch_size, parent=solver, values=["sgd", "adam"])

        # We can also add multiple conditions on hyperparameters at once:
        cs.add_conditions([use_lr, use_batch_size, use_lr_init])

        return cs

def train(config, budget_type, seed=0, budget=25):
    # For deactivated parameters (by virtue of the conditions),
    # the configuration stores None-values.
    # This is not accepted by the MLP, so we replace them with placeholder values.
    lr = config.get("learning_rate", "constant")
    lr_init = config.get("learning_rate_init", 0.001)
    batch_size = config.get("batch_size", 200)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        classifier = MLPClassifier(
            hidden_layer_sizes=[config["n_neurons"]] * config["n_layer"],
            solver=config["solver"],
            batch_size=batch_size,
            activation=config["activation"],
            learning_rate=lr,
            learning_rate_init=lr_init,
            max_iter=int(np.ceil(budget)),
            random_state=seed,
        )

        if budget_type == 'cv_splits':
            cv = StratifiedKFold(n_splits=int(np.ceil(budget)), random_state=seed, shuffle=True)
        else:
            cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

        if budget_type == 'sampling_ratio':
            sampled_data, sampled_target = random_subset(dataset, int(np.ceil(budget)))
            score = cross_val_score(classifier, sampled_data, sampled_target, cv=cv, error_score="raise")
        else:
            score = cross_val_score(classifier, dataset.data, dataset.target, cv=cv, error_score="raise")

    return 1 - np.mean(score)

def optimize_for_budget_type(smac, budget_type):
    incumbent = smac.optimize()

    default_cost = smac.validate(mlp.configspace.get_default_configuration())
    print(f"Default cost ({smac.intensifier.__class__.__name__}): {default_cost}")

    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost ({smac.intensifier.__class__.__name__}): {incumbent_cost}")

    return incumbent

def plot_trajectory(results_per_seed: dict):
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")
    plt.ylabel(next(iter(results_per_seed.values())).scenario.objectives)
    plt.ylim(0, 1)

    for (seed, budget_type), facade in results_per_seed.items():
        X, Y = [], []
        for item in facade.intensifier.trajectory:
            # Single-objective optimization
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            y = item.costs[0]
            x = item.walltime

            X.append(x)
            Y.append(y)

        plt.plot(X, Y, label=f"{budget_type} - Seed {seed}")  # Include seed in label
        plt.scatter(X, Y, marker="x")

    plt.legend()
    plt.show()

def get_best_fidelity(results_per_seed: dict) -> str:
    seperate_run_scores = {}
    for (seed, budget_type), facade in results_per_seed.items():
        X, Y = [], []
        for item in facade.intensifier.trajectory:
            # Single-objective optimization
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            y = item.costs[0]
            x = item.walltime

            X.append(x)
            Y.append(y)
            print(f"Y: {Y}, cost items for seed {seed} and budget {budget_type}")
        seperate_run_scores[(seed, budget_type)] = (np.mean(X), np.mean(Y))  # calc the mean cost and mean walltime for each seed and budget

    # avg results per seed
    # Initialize a defaultdict to store scores per seed and hyperparameter
    scores_per_seed_hyperparam = defaultdict(list)

    # Iterate through the input dictionary and group scores by seed and hyperparameter
    for (seed, hyperparam), (mean_x, mean_y) in seperate_run_scores.items():
        scores_per_seed_hyperparam[hyperparam].append((mean_x, mean_y))



    avg_dist_per_fidelity = {}

    # avg the results of the seeds per fidelity and calc the distance of the mean scores per fidelity to the origin
    for fidelity, tuples_list in scores_per_seed_hyperparam.items():
        avg_x = np.mean([t[0] for t in tuples_list])
        avg_y = np.mean([t[1] for t in tuples_list])
        avg_dist_per_fidelity[fidelity] = math.sqrt(avg_x**2 + avg_y**2)

    print("avg_dist_per_fidelity", avg_dist_per_fidelity)

    min_distance_fidelity = min(avg_dist_per_fidelity, key=avg_dist_per_fidelity.get)
    return min_distance_fidelity

if __name__ == "__main__":
    mlp = MLP()

    intensifier_object = Hyperband
    budget_types = ["epoch", "sampling_ratio"]
    mins_and_maxs = [(5, 25), (20, 100)]
    results_per_seed = {}
    for seed in range(2):
        for budget_type, (min_budget, max_budget) in zip(budget_types, mins_and_maxs):
            facades = []
            # Run 5 times with different seeds
            print(f"Budget type: {budget_type} - Seed: {seed}")

            scenario = Scenario(
                mlp.configspace,
                walltime_limit=100,
                n_trials=100,
                min_budget=min_budget,
                max_budget=max_budget,
                n_workers=8,
                seed=seed,  # Use a different seed for each run
                name=f"MLPRunBudget({budget_type}_Seed{seed})"
            )

            initial_design = MFFacade.get_initial_design(scenario, n_configs=5)
            intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")

            smac = MFFacade(
                scenario,
                partial(train, budget_type=budget_type),
                initial_design=initial_design,
                intensifier=intensifier,
                overwrite=True,
            )

            optimize_for_budget_type(smac, budget_type)
            results_per_seed[(seed, budget_type)] = smac


    logging.info(results_per_seed)
    #plot_trajectory(results_per_seed)

    print('#' * 50)

    best_fidelity = get_best_fidelity(results_per_seed)
    print("best fidelity: ", best_fidelity)

    print('#' * 50)