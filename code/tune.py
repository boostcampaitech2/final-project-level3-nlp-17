import optuna
import wandb
import yaml

from dataset import TabularDataset, TabularDatasetFromHuggingface
from src.modules import TabNet
from train import trainer

from typing import Any, Dict, List, Tuple

from transformers import HfArgumentParser
from datasets import load_dataset

from arguments import ModelArguments, DataArguments

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


import os

import pprint

BEST_MODEL_PATH = "./src/model/best_model/"


def get_best_trial_with_condition(optuna_study: optuna.study.Study) -> Dict[str, Any]:
    """Get best trial that satisfies the minimum condition(e.g. accuracy > 0.8).
    Args:
        study : Optuna study object to get trial.
    Returns:
        best_trial : Best trial that satisfies condition.
    """
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "acc_percent",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )
    ## minimum condition : accuracy >= threshold
    threshold = 0.65
    minimum_cond = df.acc_percent >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.loc[
            df_min_cond.params_nums == df_min_cond.params_nums.min()
        ].acc_percent.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_


# input_dim, output_dim, n_d, n_a, n_steps, gamma, n_independent, n_shared, virtual_batch_size, momentum, epsilon=1e-15
def search_model(trial, model_args, data_args):
    model_config = {}
    model_config["input_dim"] = model_args.input_dim
    model_config["output_dim"] = model_args.output_dim
    model_config["n_d"] = trial.suggest_int("n_d", low=2, high=64, step=2)
    model_config["n_a"] = trial.suggest_int("n_a", low=2, high=64, step=2)
    model_config["n_steps"] = trial.suggest_int("n_steps", low=1, high=32, step=1)
    model_config["gamma"] = trial.suggest_float("gamma", low=1.0, high=3.0, step=0.5)
    model_config["n_independent"] = trial.suggest_int(
        "n_independent", low=1, high=8, step=1
    )
    model_config["n_shared"] = trial.suggest_int("n_shared", low=0, high=8, step=1)
    model_config["virtual_batch_size"] = trial.suggest_int(
        "virtual_batch_size", low=64, high=512, step=2
    )
    model_config["momentum"] = trial.suggest_float(
        "momentum", low=0.1, high=0.9, step=0.1
    )
    model_config["epsilon"] = 1e-15

    model_config["cat_idxs"] = []
    model_config["cat_dims"] = []
    model_config["cat_emb_dim"] = model_args.cat_emb_dim

    return model_config


# learning_rate, epochs, l_sparse, batch_size
def search_hyperparam(trial):
    hyperparams = {}
    hyperparams["learning_rate"] = trial.suggest_float(
        "learning_rate", low=0.01, high=0.09, step=0.01
    )
    hyperparams["epochs"] = trial.suggest_int("epochs", low=50, high=100, step=10)
    hyperparams["l_sparse"] = trial.suggest_float(
        "l_sparse", low=0.00001, high=0.001, step=0.00001
    )
    hyperparams["batch_size"] = trial.suggest_int(
        "batch_size", low=512, high=2048, step=512
    )
    hyperparams["weight_decay_rate"] = trial.suggest_float(
        "weight_decay_rate", low=0.0, high=1.0, step=0.1
    )
    return hyperparams


def objective(trial, train_dataloader, val_dataloader, model_args, data_args, device):

    model_config = search_model(trial, model_args, data_args)
    hyperparams = search_hyperparam(trial)

    print(model_config)
    print(hyperparams)

    model = TabNet(**model_config).to(device)

    wandb.init(
        project="final0",
        entity="geup",
        config={"model_config": model_config, "data_config": hyperparams},
        reinit=True,
    )
    wandb.watch(model, log="all")

    loss, acc, train_acc = trainer(
        model, train_dataloader, val_dataloader, device, **hyperparams
    )

    model_config["learning_rate"] = hyperparams["learning_rate"]
    model_config["epochs"] = hyperparams["epochs"]
    model_config["l_sparse"] = hyperparams["l_sparse"]
    model_config["batch_size"] = hyperparams["batch_size"]

    if os.path.exists(os.path.join(BEST_MODEL_PATH, "best_score.txt")):
        with open(os.path.join(BEST_MODEL_PATH, "best_score.txt"), "rt") as f:
            best_score = float(f.read())
    else:
        best_score = 0

    cur_score = float(acc + train_acc)

    if cur_score > best_score:
        with open(os.path.join(BEST_MODEL_PATH, "best_model.yml"), "w") as f:
            yaml.dump(model_config, f, default_flow_style=False)

        torch.save(model.state_dict(), f=os.path.join(BEST_MODEL_PATH, "best_model.pt"))

        best_score = cur_score
        with open(os.path.join(BEST_MODEL_PATH, "best_score.txt"), "wt") as f:
            f.write(str(best_score))
        print("update best model")
    wandb.log({"loss": loss, "acc": acc})
    return acc, loss, train_acc


def tune(model_args, data_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rdb_storage = optuna.storages.RDBStorage(
        url="postgresql://optuna:0000@127.0.0.1:5432/optuna"
    )

    os.makedirs(BEST_MODEL_PATH, exist_ok=True)

    # dataset = TabularDataset(model_args, data_args, is_train=True)

    # train_len =int(len(dataset)*0.8)
    # val_len = len(dataset)-train_len
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    data_files = {"train": "train.csv", "validation": "validation.csv"}
    dataset = load_dataset("PDJ107/riot-data", data_files=data_files, revision="cgm_20")

    train_dataset = TabularDatasetFromHuggingface(dataset["train"])
    val_dataset = TabularDatasetFromHuggingface(dataset["validation"])

    print("train data len : ", len(train_dataset))
    print("validation data len : ", len(val_dataset))

    train_dataloader = DataLoader(
        train_dataset, batch_size=model_args.batch_size, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=model_args.batch_size, pin_memory=True
    )

    study = optuna.create_study(
        directions=["maximize", "minimize", "maximize"],
        study_name="final_0",  # final_s4
        sampler=optuna.samplers.MOTPESampler(),
        storage=rdb_storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial, train_dataloader, val_dataloader, model_args, data_args, device
        ),
        n_trials=500,
    )

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)
    print(best_trial)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    tune(model_args, data_args)
