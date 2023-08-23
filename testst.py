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
from smac.intensifier.hyperband import Hyperband

dataset = load_digits()

def sample_data(data, sampling_ratio):
    num_samples = int(np.ceil(data.shape[0] * sampling_ratio))
    sample_indices = np.random.choice(data.shape[0], num_samples, replace=False)
    sampled_data = data[sample_indices]
    return sampled_data

class MLP:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        n_layer = Integer("n_layer", (1, 5), default=1)
        n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
        activation = Categorical("activation", ["logistic", "tanh", "relu"], default="tanh")
        solver = Categorical("solver", ["lbfgs", "sgd", "adam"], default="adam")
        batch_size = Integer("batch_size", (30, 300), default=200)
        learning_rate = Categorical("learning_rate", ["constant", "invscaling", "adaptive"], default="constant")
        learning_rate_init = Float("learning_rate_init", (0.0001, 1.0), default=0.001, log=True)

        cs.add_hyperparameters([n_layer, n_neurons, activation, solver, batch_size, learning_rate, learning_rate_init])

        use_lr = EqualsCondition(child=learning_rate, parent=solver, value="sgd")
        use_lr_init = InCondition(child=learning_rate_init, parent=solver, values=["sgd", "adam"])
        use_batch_size = InCondition(child=batch_size, parent=solver, values=["sgd", "adam"])

        cs.add_conditions([use_lr, use_batch_size, use_lr_init])

        return cs

    def train(self, config: Configuration, budget_type: str, budget: int, seed: int = 0) -> float:
        lr = config.get("learning_rate", "constant")
        lr_init = config.get("learning_rate_init", 0.001)
        batch_size = config.get("batch_size", 200)
        
        if budget_type == "epoch":
            max_iter = budget
            n_splits = 5
            sampling_ratio = 1.0
        elif budget_type == "cv_splits":
            max_iter = 25
            n_splits = budget
            sampling_ratio = 1.0
        elif budget_type == "sampling_ratio":
            max_iter = 25
            n_splits = 5
            sampling_ratio = budget
        else:
            raise ValueError("Invalid budget_type")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            classifier = MLPClassifier(
                hidden_layer_sizes=[config["n_neurons"]] * config["n_layer"],
                solver=config["solver"],
                batch_size=batch_size,
                activation=config["activation"],
                learning_rate=lr,
                learning_rate_init=lr_init,
                max_iter=int(np.ceil(max_iter)),
                random_state=seed,
            )

            cv = StratifiedKFold(n_splits=int(np.ceil(n_splits)), random_state=seed, shuffle=True)
            sampled_data = sample_data(dataset, float(np.ceil(sampling_ratio)))
            score = cross_val_score(classifier, sample_data(dataset, float(np.ceil(sampling_ratio))).data, sample_data(dataset, float(np.ceil(sampling_ratio))).target, cv=cv, error_score="raise")

        return 1 - np.mean(score)

def plot_trajectory(facades):
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")
    plt.ylabel(facades[0].scenario.objectives)
    plt.ylim(0, 0.4)

    for facade in facades:
        X, Y = [], []
        for item in facade.intensifier.trajectory:
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

if __name__ == "__main__":
    budget_types = ["epoch", "cv_splits", "sampling_ratio"]
    budget_ranges = {
        "epoch": (5, 20),
        "cv_splits": (3, 10),
        "sampling_ratio": (0.1, 1.0)
    }
    
    best_budget_type = None
    best_reduction = float("inf")

    mlp = MLP()

    default_config = mlp.configspace.get_default_configuration()
    default_cost = mlp.train(default_config, budget_type="epoch", budget=25)


    for budget_type in budget_types:
        min_budget, max_budget = budget_ranges[budget_type]
        
        facades = []
        intensifier_object = Hyperband

        scenario = Scenario(
            mlp.configspace,
            walltime_limit=60,
            n_trials=500,
            min_budget=min_budget,
            max_budget=max_budget,
            n_workers=8,
        )
        
        initial_design = MFFacade.get_initial_design(scenario, n_configs=5)
        intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")
        
        smac = MFFacade(
            scenario,
            mlp.train,
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )
        
        incumbent = smac.optimize()
        incumbent_cost = smac.validate(incumbent)
        facades.append(smac)
        plot_trajectory(facades)
        
        reduction = default_cost - incumbent_cost
        
        if reduction < best_reduction:
            best_reduction = reduction
            best_budget_type = budget_type

    print(f"Best performing budget type: {best_budget_type}")
