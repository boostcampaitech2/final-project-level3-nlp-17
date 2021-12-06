from torch.utils.data import Dataset
import torch

from tqdm import tqdm

from transformers import HfArgumentParser
from arguments import DataArguments, ModelArguments

class TabularDataset(Dataset):
    def __init__(self, data_args):
        with open(data_args.data_path, 'r', encoding='UTF8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.tensor(list(map(int, self.data[idx].split(','))), dtype=torch.float32, requires_grad=True)
        x, label = data[:10], data[10]
        return x, label.long()

if __name__=='__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    _, data_args = parser.parse_args_into_dataclasses()
    dataset = TabularDataset(data_args)
    d = {}
    for x, label in tqdm(dataset):
        if int(label) in d:
            d[int(label)] += 1
        else:
            d[int(label)] = 1
    print(d)       