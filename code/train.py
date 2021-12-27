from src.modules import TabNetNoEmbeddings, TabNet, TabNetPretraining
from transformers import HfArgumentParser
from datasets import load_dataset
from arguments import ModelArguments, DataArguments
from dataset import (
    TabularDataset,
    EasyTabularDataset,
    TabularDatasetFromHuggingface,
    construct_loaders,
)

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import torch.nn as nn
import torch.optim as optim

from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score

from tqdm import tqdm

import wandb

import numpy as np

import os

import random
import torch.backends.cudnn as cudnn

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import check_array

import copy


def seed_fix():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.set_default_dtype(torch.float32)


def evaluation(model, val_dataloader, criterion, l_sparse, device):
    model.eval()

    list_y_true = []
    list_y_score = []

    with torch.no_grad():
        for x, label in tqdm(val_dataloader):
            logits, _ = model(x.to(device))

            logits = logits.cpu().detach().numpy()
            list_y_true.append(label)
            list_y_score.append(logits)

    y_true = np.hstack(list_y_true)
    y_score = np.vstack(list_y_score)
    y_score = softmax(y_score, axis=1)

    y_pred = np.argmax(y_score, axis=1)
    val_acc = accuracy_score(y_true, y_pred)

    print("val_acc : ", val_acc)
    wandb.log({"val_acc": val_acc})
    model.train()
    return val_acc


def TabNetSelfLoss(y_pred, embedded_x, obf_vars, eps=1e-9):
    """
    Implements unsupervised loss function.
    This differs from orginal paper as it's scaled to be batch size independent
    and number of features reconstructed independent (by taking the mean)
    """
    errors = y_pred - embedded_x
    reconstruction_errors = torch.mul(errors, obf_vars) ** 2
    batch_stds = torch.std(embedded_x, dim=0) ** 2 + eps
    features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
    # take the mean of the reconstructed variable errors
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    # here we take the mean per batch, contrary to the paper
    loss = torch.mean(features_loss)
    return loss


def self_train(model, train_dataloader, val_dataloader, model_args, data_args, device):

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=model_args.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.95 ** (epoch // 10),
        last_epoch=-1,
    )

    criterion = TabNetSelfLoss
    for _ in range(model_args.self_supervised_learning_epochs):
        for epoch in range(model_args.epochs):
            running_loss = 0.0
            val_len = 0.0

            with tqdm(train_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"[epoch : {epoch}]")
                for x, _ in tepoch:
                    optimizer.zero_grad()
                    x = x.to(device)
                    logits, embedded_x, obf_vars = model(x)
                    label = torch.mul(x, obf_vars)

                    loss = criterion(logits, embedded_x, obf_vars)

                    loss.backward(retain_graph=True)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                    optimizer.step()

                    running_loss += loss.detach().cpu()
                    val_len += 1.0

                    # wandb.log({"train_accuary": train_acc, "train_loss": train_loss})
                    tepoch.set_postfix(loss=f"{running_loss.item()/val_len:.3f}")

            scheduler.step()


def train(model, train_dataloader, val_dataloader, model_args, data_args, device):

    model.train()

    optimizer = optim.Adam(
        model.parameters(),
        lr=model_args.learning_rate,
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=lambda epoch: 0.9 ** (epoch // 10), last_epoch=-1
    )

    criterion = nn.functional.cross_entropy

    # evaluation(model, val_dataloader, criterion, model_args.l_sparse, device)

    with open("model.txt", "w") as f:
        for param in model.parameters():
            f.write(str(param))

    for epoch in range(model_args.epochs):
        running_acc = 0.0
        running_loss = 0.0
        val_len = 0.0
        val_num = 0.0

        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"[epoch : {epoch}]")

            for batch_idx, (x, label) in enumerate(tepoch):

                x = x.to(device).float()
                label = label.to(device).float()

                optimizer.zero_grad()

                logits, M_loss = model(x)

                loss = criterion(logits, label.long())

                loss = loss - model_args.l_sparse * M_loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()

                preds = torch.argmax(logits.detach().cpu(), dim=1)
                acc = torch.sum(preds == label.detach().cpu())

                val_num += float(x.shape[0])
                val_len += 1.0
                running_loss += loss.detach().cpu()
                running_acc += acc

                tepoch.set_postfix(
                    loss=f"{running_loss.item()/val_len:.3f}",
                    acc=f"{running_acc/val_num:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                )

        scheduler.step()
        wandb.log({"train_acc": running_acc / val_num})
        val_acc = evaluation(
            model, val_dataloader, criterion, model_args.l_sparse, device
        )


def trainer(
    model,
    train_dataloader,
    val_dataloader,
    device,
    learning_rate,
    epochs,
    l_sparse,
    batch_size,
    weight_decay_rate,
):

    model.train()

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.95 ** (epoch // 10),
        last_epoch=-1,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_acc = 0.0
        running_loss = 0.0
        val_len = 0.0
        val_num = 0.0

        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"[epoch : {epoch}]")
            for x, label in tepoch:
                optimizer.zero_grad()
                logits, M_loss = model(x.to(device))

                loss = criterion(logits, label.to(device)) - l_sparse * M_loss

                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()

                preds = torch.argmax(logits.detach().cpu(), dim=1)
                acc = torch.sum(preds == label.detach().cpu())

                val_num += float(x.shape[0])
                val_len += 1.0
                running_loss += loss.detach().cpu()
                running_acc += acc

                tepoch.set_postfix(
                    loss=f"{running_loss.item()/val_len:.3f}",
                    acc=f"{running_acc/val_num:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                )

        wandb.log({"train_acc": running_acc / val_num})
        val_acc = evaluation(model, val_dataloader, criterion, l_sparse, device)
        scheduler.step()
    return val_acc, running_acc / val_num


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


if __name__ == "__main__":
    seed_fix()

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="final_use_sampler",
        entity="geup",
        config={"model_config": model_args, "data_config": data_args},
        reinit=True,
    )

    data_files = {"train": "train.csv", "validation": "validation.csv"}
    dataset = load_dataset("PDJ107/riot-data", data_files=data_files, revision="cgm_20")

    train_dataset = TabularDatasetFromHuggingface(dataset["train"])
    val_dataset = TabularDatasetFromHuggingface(dataset["validation"])

    train_dataloader, valid_dataloader = construct_loaders(
        train_dataset,
        val_dataset,
        weights=1,  # 1 : use weighted random sampler, 0 : not use weighted random sampler
        device=device,
    )

    model = TabNet(
        model_args.input_dim,
        model_args.output_dim,
        model_args.n_d,
        model_args.n_a,
        model_args.n_steps,
        model_args.gamma,
        train_dataset.cat_idxs,
        train_dataset.cat_dims,
        model_args.cat_emb_dim,
        model_args.n_independent,
        model_args.n_shared,
        model_args.virtual_batch_size,
        model_args.momentum,
        model_args.epsilon,
    ).to(device)

    if model_args.is_pretrain:

        self_model = TabNetPretraining(
            model_args.input_dim,
            model_args.output_dim,
            model_args.n_d,
            model_args.n_a,
            model_args.n_steps,
            model_args.gamma,
            train_dataset.cat_idxs,
            train_dataset.cat_dims,
            model_args.cat_emb_dim,
            model_args.n_independent,
            model_args.n_shared,
            model_args.virtual_batch_size,
            model_args.momentum,
            model_args.epsilon,
        ).to(device)
        print("start pretraining")
        self_train(
            self_model,
            train_dataloader,
            valid_dataloader,
            model_args,
            data_args,
            device,
        )
        model.load_state_dict(self_model.state_dict(), strict=False)

    print(get_n_params(model))
    wandb.watch(model, log="all")

    print("start classification learning")
    train(model, train_dataloader, valid_dataloader, model_args, data_args, device)
    torch.save(model.state_dict(), f="./src/model/model.pt")
