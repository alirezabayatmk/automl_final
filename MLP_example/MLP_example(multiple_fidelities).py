import warnings

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

    def train_epoch(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
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

            # Returns the 5-fold cross validation accuracy
            cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)  # to make CV splits consistent
            score = cross_val_score(classifier, dataset.data, dataset.target, cv=cv, error_score="raise")

        return 1 - np.mean(score)


    def train_cv(self, config: Configuration, seed: int = 0, budget: int = 10) -> float:
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
                max_iter=25,
                random_state=seed,
            )

            # Returns the 5-fold cross validation accuracy
            cv = StratifiedKFold(n_splits=int(np.ceil(budget)), random_state=seed, shuffle=True)  # to make CV splits consistent
            score = cross_val_score(classifier, dataset.data, dataset.target, cv=cv, error_score="raise")

        return 1 - np.mean(score)


    def train_subset(self, config: Configuration, seed: int = 0, budget: int = 100) -> float:
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
                max_iter=25,
                random_state=seed,
            )

            # Returns the 5-fold cross validation accuracy
            cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)  # to make CV splits consistent
            sampled_data, sampled_target = random_subset(dataset, int(np.ceil(budget)))
            score = cross_val_score(classifier, sampled_data, sampled_target, cv=cv, error_score="raise")

        return 1 - np.mean(score)



def plot_trajectory(facades: list[AbstractFacade], budget_types: list[str]) -> None:
    """Plots the trajectory (incumbents) of the optimization process."""
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")
    plt.ylabel(facades[0].scenario.objectives)
    plt.ylim(0, 0.4)

    for facade, budget_type in zip(facades, budget_types):
        X, Y = [], []
        for item in facade.intensifier.trajectory:
            # Single-objective optimization
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            y = item.costs[0]
            x = item.walltime

            X.append(x)
            Y.append(y)

        plt.plot(X, Y, label=budget_type)  # Use budget_type as the label
        plt.scatter(X, Y, marker="x")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    mlp = MLP()

    facades: list[AbstractFacade] = []

    intensifier_object = Hyperband

    budget_types = ["epoch", "cv_splits", "sampling_ratio"]
    mins_and_maxs = [(5, 25), (1, 10), (20, 100)]

    for budget_type, (min_budget, max_budget) in zip(budget_types, mins_and_maxs):
        print(f"Budget type: {budget_type}")
    
        scenario = Scenario(
            mlp.configspace,
            walltime_limit=360,  
            n_trials=500,  
            min_budget= min_budget,  
            max_budget= max_budget, 
            n_workers=16,
            name="MLPRunBudget({})".format(budget_type)
        )

        initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

        intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")

        if budget_type=='epoch':
            smac = MFFacade(
                scenario,
                mlp.train_epoch,
                initial_design=initial_design,
                intensifier=intensifier,
                overwrite=True,
            )

        elif budget_type=='cv_splits':
            smac = MFFacade(
                scenario,
                mlp.train_cv,
                initial_design=initial_design,
                intensifier=intensifier,
                overwrite=True,
            )

        elif budget_type=='sampling_ratio':
            smac = MFFacade(
                scenario,
                mlp.train_subset,
                initial_design=initial_design,
                intensifier=intensifier,
                overwrite=True,
            )

        incumbent = smac.optimize()

        default_cost = smac.validate(mlp.configspace.get_default_configuration())
        print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")

        incumbent_cost = smac.validate(incumbent)
        print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")

        facades.append(smac)

        plot_trajectory(facades, budget_types)

        print('#'*50)
