from src.modules import TabNetNoEmbeddings, TabNet, TabNetPretraining
from transformers import HfArgumentParser
from datasets import load_dataset
from arguments import (ModelArguments, DataArguments)
from dataset import TabularDataset, EasyTabularDataset, TabularDatasetFromHuggingface

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score

from tqdm import tqdm

import wandb

import numpy as np

import os


def evaluation(model, val_dataloader, criterion, l_sparse, device):
    model.eval()
    running_loss = 0
    running_acc = 0
    running_f1 = 0
    val_len = 0

    for x, label in tqdm(val_dataloader):
        logits, M_loss  = model(x.to(device))

        loss = criterion(logits, label.to(device)) + l_sparse * M_loss
        preds = torch.argmax(logits.detach().cpu(), dim=1)
        running_loss += loss.detach().cpu() * float(x.shape[0])
        running_acc += torch.sum(preds == label.detach().cpu())
        running_f1 += f1_score(label.numpy(), preds.numpy(), average="micro") * float(x.shape[0])
        val_len += float(x.shape[0])

    val_loss = float(running_loss) / val_len
    val_acc = float(running_acc) / val_len
    val_f1 = float(running_f1) / val_len
    
    print("val_loss : ", val_loss, "val_acc : ", val_acc, "val_f1 : ", val_f1)
    wandb.log({"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})
    model.train()
    return val_loss, val_acc, val_f1

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
        lr_lambda = lambda epoch: 0.95 ** (epoch//10),
        last_epoch=-1
        )

    criterion = TabNetSelfLoss
    for _ in range(model_args.self_supervised_learning_epochs):
        for epoch in range(model_args.epochs):
            running_loss = 0.
            val_len = 0.

            with tqdm(train_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"[epoch : {epoch}]")
                for x, _ in tepoch:
                    optimizer.zero_grad()
                    x = x.to(device)
                    logits, embedded_x, obf_vars = model(x)
                    label = torch.mul(x, obf_vars)

                    loss = criterion(logits, embedded_x, obf_vars)

                    loss.backward(retain_graph=True)
                    optimizer.step()

                    running_loss += loss.detach().cpu()
                    val_len += 1.

                    #wandb.log({"train_accuary": train_acc, "train_loss": train_loss})
                    tepoch.set_postfix(loss=f"{running_loss.item()/val_len:.3f}")

def train(model, train_dataloader, val_dataloader, model_args, data_args, device):
    

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=model_args.learning_rate, weight_decay=model_args.weight_decay_rate)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda = lambda epoch: 0.95 ** epoch,
        last_epoch=-1
        )

    criterion = nn.CrossEntropyLoss()

    #evaluation(model, val_dataloader, criterion, model_args.l_sparse, device)

    for epoch in range(model_args.epochs):
        running_acc = 0.
        running_loss = 0.
        val_len = 0.
        val_num = 0.

        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"[epoch : {epoch}]")

            for x, label in tepoch:

                optimizer.zero_grad()
                logits, M_loss = model(x.to(device))

                loss = criterion(logits, label.to(device)) + model_args.l_sparse * M_loss


                loss.backward(retain_graph=True)

                optimizer.step()
                scheduler.step()

                preds = torch.argmax(logits.detach().cpu(), dim=1)
                acc = torch.sum(preds == label.detach().cpu())

                val_num += float(x.shape[0])
                val_len += 1.
                running_loss += loss.detach().cpu()
                running_acc += acc
                
                tepoch.set_postfix(
                    loss=f"{running_loss.item()/val_len:.3f}", acc=f"{running_acc/val_num:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}"
                )
        #wandb.log({"train_acc": running_acc/val_num})
        val_loss, val_acc, val_f1 = evaluation(model, val_dataloader, criterion, model_args.l_sparse, device)

    
def trainer(model, train_dataloader, val_dataloader, device, learning_rate, epochs, l_sparse, batch_size, weight_decay_rate):

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay_rate)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda = lambda epoch: 0.95 ** (epoch//10),
        last_epoch=-1
        )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_acc = 0.
        running_loss = 0.
        val_len = 0.
        val_num = 0.

        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"[epoch : {epoch}]")
            for x, label in tepoch:
                optimizer.zero_grad()
                logits, M_loss = model(x.to(device))

                loss = criterion(logits, label.to(device)) + l_sparse * M_loss

                loss.backward(retain_graph=True)
                optimizer.step()
                

                preds = torch.argmax(logits.detach().cpu(), dim=1)
                acc = torch.sum(preds == label.detach().cpu())

                val_num += float(x.shape[0])
                val_len += 1.
                running_loss += loss.detach().cpu()
                running_acc += acc
                
            
                tepoch.set_postfix(
                    loss=f"{running_loss.item()/val_len:.3f}", acc=f"{running_acc/val_num:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}"
                )
            
        wandb.log({"train_acc": running_acc/val_num})
        val_loss, val_acc, val_f1 = evaluation(model, val_dataloader, criterion, l_sparse, device)
        scheduler.step()
    return val_loss, val_acc, running_acc/val_num



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

if __name__=='__main__':

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' ) #'cuda' if torch.cuda.is_available() else 'cpu'

    # wandb.init(
    #     project="final",
    #     entity='geup',
    #     config={'model_config':model_args, 'data_config':data_args},
    #     reinit = True
    # )

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

    self_model = TabNetPretraining(
        model_args.input_dim,
        model_args.pretraining_ratio,
        model_args.n_d,
        model_args.n_a,
        model_args.n_steps,
        model_args.gamma,
        [],
        [],
        model_args.cat_emb_dim,
        model_args.n_independent,
        model_args.n_shared,
        model_args.epsilon,
        model_args.virtual_batch_size,
        model_args.momentum,
        model_args.n_shared_decoder,
        model_args.n_indep_decoder,
        ).to(device) 

    # wandb.watch(model, log="all")

    data_files = {"train": "train.csv", "validation": "validation.csv"}
    dataset = load_dataset("PDJ107/riot-data", data_files=data_files, revision='cgm_20')

    train_dataset = TabularDatasetFromHuggingface(dataset['train'])
    val_dataset = TabularDatasetFromHuggingface(dataset['validation'])

    # dataset = TabularDataset(model_args, data_args, is_train=True)

    # train_len =int(len(dataset)*0.8)
    # val_len = len(dataset)-train_len
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])



    print('train data len : ', len(train_dataset))
    print('validation data len : ', len(val_dataset))


    train_dataloader = DataLoader(train_dataset, batch_size=model_args.batch_size, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=model_args.batch_size, pin_memory=True)

    # easy_dataset = EasyTabularDataset(model_args, data_args, is_train=True)

    # easy_train_len =int(len(easy_dataset)*0.8)
    # easy_val_len = len(easy_dataset)-train_len
    # easy_train_dataset, easy_val_dataset = torch.utils.data.random_split(easy_dataset, [easy_train_len, easy_val_len])

    # easy_train_dataloader = DataLoader(easy_train_dataset, batch_size=model_args.batch_size, pin_memory=True)
    # easy_val_dataloader = DataLoader(easy_val_dataset, batch_size=model_args.batch_size, pin_memory=True)

    
    print(get_n_params(model))

        
    if os.path.exists('./src/model/pretrain_model.pt'):
        model.load_state_dict(torch.load('./src/model/model.pt', map_location=device))
    else:
        print('start self supervised learning')
        self_train(self_model, train_dataloader, val_dataloader, model_args, data_args, device)
        model.load_state_dict(self_model.state_dict(), strict=False)
        # print('start easy train')
        # train(model, easy_train_dataloader, easy_val_dataloader, model_args, data_args, device)
        # torch.save(model.state_dict(), f='./src/model/pretrain_model.pt')

    print('start classification learning')
    train(model, train_dataloader, val_dataloader, model_args, data_args, device)
    torch.save(model.state_dict(), f='./src/model/model.pt')



    