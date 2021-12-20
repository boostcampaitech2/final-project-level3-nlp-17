import torch

from src.modules import TabNet
from dataset import InnerEvalDataset

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

import pandas as pd

import yaml

import os

BEST_MODEL_PATH = './src/model/best_model/'

if __name__=='__main__':

    a_team_data_path = '../data/dataset/DATA_CHALLENGER_I_MatchData_VER6.csv'
    a_team_label_path = '../data/dataset/LABEL_CHALLENGER_I_MatchData_VER6.csv'
    b_team_data_path = '../data/dataset/DATA_IRON_IV_MatchData_VER6.csv'
    b_team_label_path = '../data/dataset/LABEL_IRON_IV_MatchData_VER6.csv'

    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' ) #'cuda' if torch.cuda.is_available() else 'cpu'


    with open(os.path.join(BEST_MODEL_PATH, "best_model.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config['input_dim'])
    
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

    dataset = InnerEvalDataset(a_team_data_path, a_team_label_path, b_team_data_path, b_team_label_path)

    alar = 0.
    albr = 0.
    blar = 0.
    blbr = 0.

    data_len = float(len(dataset))

    softmax = nn.Softmax(dim=-1)
    
    i = 0
    i_max = 100
    for alar_data, albr_data, blar_data, blbr_data in tqdm(dataset):

        alar_output, _ = model(alar_data.view(1,-1).to(device))
        alar_p_output = softmax(alar_output.detach().cpu())[0][0]
        
        albr_output, _ = model(albr_data.view(1,-1).to(device))
        albr_p_output = softmax(albr_output.detach().cpu())[0][0]

        blar_output, _ = model(blar_data.view(1,-1).to(device))
        blar_p_output = softmax(blar_output.detach().cpu())[0][0]

        blbr_output, _ = model(blbr_data.view(1,-1).to(device))
        blbr_p_output = softmax(blbr_output.detach().cpu())[0][0]

        alar += alar_p_output/i_max
        albr += albr_p_output/i_max
        blar += blar_p_output/i_max
        blbr += blbr_p_output/i_max

        i+=1
        if i>=i_max:
            break

    print(alar, '  |  ', albr)
    print(blar, '  |  ', blar)
        

    