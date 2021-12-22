import torch

from src.modules import TabNetNoEmbeddings, TabNet, TabNetPretraining
from transformers import HfArgumentParser
from datasets import load_dataset
from arguments import (ModelArguments, DataArguments)
from dataset import TabularDataset, TabularDatasetFromHuggingface

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

import pandas as pd


if __name__=='__main__':

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' ) #'cuda' if torch.cuda.is_available() else 'cpu'

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

    model.load_state_dict(torch.load('./src/model/model.pt', map_location=device))

    model.eval()

    data_files = {"test": "test.csv"}
    dataset = load_dataset("PDJ107/riot-data", data_files=data_files, revision='cgm_20', use_auth_token=True)

    dataset = TabularDatasetFromHuggingface(dataset['test'])

    softmax = nn.Softmax(dim=-1)

    t = 0
    f = 0

    for x, label in tqdm(dataset):
        output, _ = model(x.view(1,-1).to(device))
        p_output = softmax(output.detach().cpu())
        max_p = torch.max(p_output)
        
        output = torch.argmax(output, dim=1)

        if output != label:
            f += 1
        else:
            t += 1
        
    print('test accuracy : ', t/(t+f+1e-10))



    

    