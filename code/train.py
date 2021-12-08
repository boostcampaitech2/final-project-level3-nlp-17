from src.modules import TabNet
from transformers import HfArgumentParser
from arguments import (ModelArguments, DataArguments)
from dataset import TabularDataset

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score

from tqdm import tqdm

def sparsity_regularization(M):
    L_sparse = 0
    eps=1e-10
    N, B, D = M.shape
    NB = N*B
    for i in range(N):
        for b in range(B):
            for j in range(D):
                L_sparse += (-M[i,b,j]*torch.log(M[i,b,j]+eps))/float(NB)
    return L_sparse

def evaluation(model, val_dataloader, criterion):
    model.eval()
    running_loss = 0
    running_acc = 0
    running_f1 = 0
    val_len = 0
    val_cnt = 0

    for x, label in tqdm(val_dataloader):
        _, logits, _, _, total_mask = model(x.float())

        loss = criterion(logits, label) + model_args.l_sparse * sparsity_regularization(total_mask)
        preds = torch.argmax(logits, dim=1)
        running_loss += loss
        running_acc += torch.sum(preds == label)
        running_f1 += f1_score(label.numpy(), preds.numpy(), average="micro")
        val_len += x.shape[0]
        val_cnt += 1

    val_loss = float(running_loss) / val_cnt
    val_acc = float(running_acc) / val_len
    val_f1 = float(running_f1) / val_cnt
    print(
        "val_loss : ",
        val_loss,
        "val_acc : ",
        val_acc,
        "val_f1 : ",
        val_f1,
    )
    #wandb.log({"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})
    model.train()

def self_train(model, model_args, data_args):
    
    dataset = TabularDataset(model_args, data_args)
    train_len =int(len(dataset)*0.8)
    val_len = len(dataset)-train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_dataloader = DataLoader(train_dataset, batch_size=data_args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=data_args.batch_size)

    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=model_args.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda = lambda epoch: 0.95 ** epoch,
        last_epoch=-1
        )

    criterion = nn.MSELoss()
    for _ in range(model_args.self_supervised_learning_epochs):
        S = torch.randint(low = 0 ,high = 2, size=(data_args.batch_size, model_args.input_dim))
        for epoch in range(model_args.epochs):
            running_acc = 0
            running_loss = 0
            train_len = 0

            with tqdm(train_dataloader, unit="batch") as tepoch:
                
                for x, _ in tepoch:
                    
                    label = x*(1-S[:x.shape[0],:])
                    optimizer.zero_grad()
                    _, _, _, logits, total_mask = model(x)

                    loss = criterion(logits, label) + model_args.l_sparse * sparsity_regularization(total_mask)

                    loss.backward(retain_graph=True)
                    optimizer.step()

                    running_loss += loss
                    train_len += 1

                    train_loss = running_loss / train_len
                    #wandb.log({"train_accuary": train_acc, "train_loss": train_loss})
                    tepoch.set_postfix(
                        loss=f"{train_loss.item():.3f}"
                    )

def train(model, model_args, data_args):
    
    dataset = TabularDataset(model_args, data_args)
    train_len =int(len(dataset)*0.8)
    val_len = len(dataset)-train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_dataloader = DataLoader(train_dataset, batch_size=data_args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=data_args.batch_size)

    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=model_args.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda = lambda epoch: 0.95 ** epoch,
        last_epoch=-1
        )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(model_args.epochs):
        running_acc = 0
        running_loss = 0
        train_len = 0
        train_cnt = 0

        with tqdm(train_dataloader, unit="batch") as tepoch:
            
            for x, label in tepoch:
                
                optimizer.zero_grad()
                _, logits, _, _, total_mask = model(x)

                loss = criterion(logits, label) + model_args.l_sparse * sparsity_regularization(total_mask)

                loss.backward(retain_graph=True)
                optimizer.step()

                preds = torch.argmax(logits, dim=1)
                running_acc += torch.sum(preds == label)
                running_loss += loss
                train_len += float(len(x))
                train_cnt += 1

                train_acc = running_acc / train_len
                train_loss = running_loss / train_cnt
                #wandb.log({"train_accuary": train_acc, "train_loss": train_loss})
                tepoch.set_postfix(
                    loss=f"{train_loss.item():.3f}", acc=f"{train_acc:.3f}"
                )
            evaluation(model, val_dataloader, criterion)




if __name__=='__main__':

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    

    model = TabNet(
        batch_size=data_args.batch_size, 
        input_dim = model_args.input_dim, 
        output_dim=model_args.output_dim,
        step_num=model_args.step_num, 
        Nd=model_args.Nd, 
        Na=model_args.Na
        )

    print('start self supervised learning')
    self_train(model, model_args, data_args)

    print('start classification learning')
    train(model, model_args, data_args)

    