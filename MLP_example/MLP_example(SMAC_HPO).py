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

import optuna


dataset = load_digits()


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

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
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


def plot_trajectory(facades: list[AbstractFacade]) -> None:
    """Plots the trajectory (incumbents) of the optimization process."""
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")
    plt.ylabel(facades[0].scenario.objectives)
    plt.ylim(0, 0.4)

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

if __name__ == "__main__":
    mlp = MLP()

    facades: list[AbstractFacade] = []

    def objective(trial):
        rf_params = optimize_hyperparameters(trial)

        scenario = Scenario(
            name="HPO-on-SMAC",
            configspace=mlp.configspace,
            walltime_limit=30,
            n_trials=50,
            min_budget=1,
            max_budget=25,
            n_workers=16,
            deterministic=True,
        )

        initial_design = MFFacade.get_initial_design(scenario, n_configs=10)
        intensifier = Hyperband(
            scenario,
            eta=3,
            incumbent_selection="highest_budget",
        )

        smac = MFFacade(
            scenario=scenario,
            target_function=mlp.train,
            initial_design=initial_design,
            intensifier=intensifier,
            model=MFFacade.get_model(scenario, **rf_params),  # Use get_model with optimized hyperparameters
            overwrite=True,
        )

        incumbent = smac.optimize()
        return smac.validate(incumbent)

    study = optuna.create_study(direction="minimize", study_name="SMAC_HPO")
    study.optimize(objective, n_trials=10, n_jobs=-1)

    best_params = study.best_params
    best_value = study.best_value
    print("Best Parameters:", best_params)
    print("Best Value:", best_value)

    with open('best_optuna_params.txt', 'w+') as f:
        f.write(str(best_params))

    best_scenario = Scenario(
        name="HPO-on-SMAC(best)",
        configspace=mlp.configspace,
        walltime_limit=100,
        n_trials=200,
        min_budget=1,
        max_budget=25,
        n_workers=16,
        deterministic=True,
    )

    best_initial_design = MFFacade.get_initial_design(best_scenario, n_configs=20)
    best_intensifier = Hyperband(
        best_scenario,
        eta=3,
        incumbent_selection="highest_budget",
    )

    best_smac = MFFacade(
        scenario=best_scenario,
        target_function=mlp.train,
        initial_design=best_initial_design,
        intensifier=best_intensifier,
        model = MFFacade.get_model(best_scenario, **best_params),  # Use get_model with optimized hyperparameters
        overwrite=True,
    )

    best_incumbent = best_smac.optimize()

    default_cost = best_smac.validate(mlp.configspace.get_default_configuration())
    print(f"Default cost ({best_intensifier.__class__.__name__}): {default_cost}")

    incumbent_cost = best_smac.validate(best_incumbent)
    print(f"Incumbent cost ({best_intensifier.__class__.__name__}): {incumbent_cost}")

    facades.append(best_smac)

    plot_trajectory(facades)

