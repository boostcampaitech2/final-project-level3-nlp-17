import torch

from src.modules import TabNetNoEmbeddings, TabNet, TabNetPretraining
from transformers import HfArgumentParser
from datasets import load_dataset
from arguments import ModelArguments, DataArguments
from dataset import TabularDataset, TabularDatasetFromHuggingface

import torch
from torch import nn
from torch.utils.data import DataLoader

from scipy.special import softmax
from sklearn.metrics import accuracy_score, r2_score

from tqdm import tqdm

import pandas as pd

import yaml

import numpy as np

if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  #'cuda' if torch.cuda.is_available() else 'cpu'

    model = TabNet(
        model_args.input_dim,
        model_args.output_dim,
        model_args.n_d,
        model_args.n_a,
        model_args.n_steps,
        model_args.gamma,
        [],
        [],
        model_args.cat_emb_dim,
        model_args.n_independent,
        model_args.n_shared,
        model_args.virtual_batch_size,
        model_args.momentum,
        model_args.epsilon,
    ).to(device)

    model.load_state_dict(torch.load("./src/model/model.pt", map_location=device))

    model.eval()

    data_files = {"test": "test.csv"}
    dataset = load_dataset(
        "PDJ107/riot-data",
        data_files=data_files,
        revision="cgm_20",
        use_auth_token=True,
    )

    dataset = TabularDatasetFromHuggingface(dataset["test"], False)

    print("test data len : ", len(dataset))

    test_dataloader = DataLoader(dataset, batch_size=1024)

    list_y_true = []
    list_y_score = []

    for x, label in tqdm(test_dataloader):
        logits, _ = model(x.to(device))

        logits = logits.cpu().detach().numpy()
        list_y_true.append(label)
        list_y_score.append(logits)
    y_true = np.hstack(list_y_true)
    y_score = np.vstack(list_y_score)
    y_score = softmax(y_score, axis=1)

    y_pred = np.argmax(y_score, axis=1)
    acc = accuracy_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    Adj_r2 = 1 - (1 - r2_score(y_true, y_pred)) * (
        (len(dataset.data) - 1) / (len(dataset.data) - len(dataset.data[0]) - 1)
    )

    print("test accuracy : ", acc)
    print("r2 : ", r2)
    print("Adj_r2 : ", Adj_r2)
