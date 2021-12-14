import torch

from src.modules import TabNetNoEmbeddings, TabNet, TabNetPretraining
from transformers import HfArgumentParser
from arguments import (ModelArguments, DataArguments)
from dataset import TabularDataset, TestTabularDataset

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

    dataset = TabularDataset(model_args, data_args, is_train=True)
    softmax = nn.Softmax(dim=-1)

    t = 0
    f = 0
    i = 0
    j = 0
    for x, label in dataset:
        output, _ = model(x.view(1,-1).to(device))
        p_output = softmax(output.detach().cpu())
        max_p = torch.max(p_output)
        
        output = torch.argmax(output, dim=1)

        #print(f'({p_output[0][0].item():.2f}, {p_output[0][1].item():.2f})', 'label : ',label.item())
        # if f >= 10:
        #     break

        if output != label:
            
            f += 1
            #print('---------False-----------')
            #print(f'({p_output[0][0].item():.2f}, {p_output[0][1].item():.2f})', 'label : ',label.item())
            # if (max_p>=0.999) and (j==0):
            #     print('---------False-----------')
            #     print(f'({p_output[0][0].item():.2f}, {p_output[0][1].item():.2f})', 'label : ',label.item())
            #     false_x = x.detach().cpu()
            #     j+=1
            #     if i+j == 2:
            #         break

            
        else:
            t += 1
            #print('---------True-----------')
            #print(f'({p_output[0][0].item():.2f}, {p_output[0][1].item():.2f})', 'label : ',label.item())
            # if (max_p>=0.999) and (i==0):
            #     print('---------True-----------')
            #     print(f'({p_output[0][0].item():.2f}, {p_output[0][1].item():.2f})', 'label : ',label.item())
            #     true_x = x.detach().cpu()
            #     i+=1
            #     if i+j == 2:
            #         break
    
    print('acc : ', t/(t+f+1e-10))
    # false_x = torch.stack([false_x[i*35:(i+1)*35] for i in range(10)], dim=0)

    # false_df = pd.DataFrame(false_x.numpy(), columns=dataset.columns.split(',')[:35])
    # false_df.to_csv('./false_data.csv')

    # true_x = torch.stack([true_x[i*35:(i+1)*35] for i in range(10)], dim=0)

    # true_df = pd.DataFrame(true_x.numpy(), columns=dataset.columns.split(',')[:35])
    # true_df.to_csv('./true_data.csv')



    

    