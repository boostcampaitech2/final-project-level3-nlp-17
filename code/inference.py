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

import yaml

if __name__=='__main__':

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' ) #'cuda' if torch.cuda.is_available() else 'cpu'

    with open('./src/model/best_model/best_model.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = TabNet(
        config['input_dim'],
        config['output_dim'],
        config['n_d'],
        config['n_a'],
        config['n_steps'],
        config['gamma'],
        config['cat_idxs'],
        config['cat_dims'],
        config['cat_emb_dim'],
        config['n_independent'],
        config['n_shared'],
        config['virtual_batch_size'],
        config['momentum'],
        config['epsilon'],
        ).to(device)

    model.load_state_dict(torch.load('./src/model/best_model/best_model.pt', map_location=device))

    model.eval()

    data_files = {"test": "test.csv"}
    dataset = load_dataset("PDJ107/riot-data", data_files=data_files, revision='cgm_20', use_auth_token=True)

    dataset = TabularDatasetFromHuggingface(dataset['test'])

    print('test data len : ', len(dataset))


    test_dataloader = DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True)

    t = 0

    for x, label in tqdm(test_dataloader):
        logits, M_loss = model(x.to(device))
        
        preds = torch.argmax(logits.detach().cpu(), dim=1)
        t += torch.sum(preds == label.detach().cpu())
        
    print('test accuracy : ', float(t/len(dataset)))



    

    