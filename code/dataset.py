from torch.utils.data import Dataset
import torch

from tqdm import tqdm

from transformers import HfArgumentParser
from arguments import DataArguments, ModelArguments

class TabularDataset(Dataset):
    def __init__(self, model_args, data_args):
        self.data_args = data_args
        self.model_args = model_args
        with open(data_args.data_path, 'r', encoding='UTF8') as f:
            self.data = f.readlines()
        self.data = self.data[1:]
        with open(data_args.label_path, 'r', encoding='UTF8') as f:
            raw_labels = f.readlines()
        
        self.label={}
        win_idx = raw_labels[0].split(',').index('_win_0_l')
        matchid_idx = raw_labels[0].split(',').index('_matchId\n')
        for raw_label in raw_labels[1:]:
            raw_label = raw_label.split(',')
            self.label[raw_label[matchid_idx]] = int(raw_label[win_idx])


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        matchId = self.data[idx].split(',')[-1]
        data = torch.tensor(list(map(float, self.data[idx].split(',')[:-1])), dtype=torch.float32, requires_grad=True)
        label = torch.tensor(self.label[matchId])

        return data, label

class TestTabularDataset(Dataset):
    def __init__(self, model_args, data_args):
        self.data_args = data_args
        self.model_args = model_args
        with open(data_args.data_path, 'r', encoding='UTF8') as f:
            self.data = f.readlines()


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        data = torch.tensor(list(map(float, self.data[idx].split(','))), dtype=torch.float32, requires_grad=True)
        x = data[:10]
        label = data[10]

        return x, label.long()

if __name__=='__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    dataset = TabularDataset(model_args, data_args)
    d = {}
    for x, label in tqdm(dataset):
        if int(label) in d:
            d[int(label)] += 1
        else:
            d[int(label)] = 1
    print(d)       