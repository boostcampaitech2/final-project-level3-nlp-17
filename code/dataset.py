from torch.utils.data import Dataset
import torch

from tqdm import tqdm

from transformers import HfArgumentParser
from arguments import DataArguments, ModelArguments
import pandas as pd


class TabularDataset(Dataset):
    def __init__(self, model_args, data_args, is_train=True):

        if is_train:
            self.data_path = data_args.data_path
            self.label_path = data_args.label_path
        else:
            self.data_path = data_args.test_data_path
            self.label_path = data_args.test_label_path

        with open(self.data_path, "r", encoding="UTF8") as f:
            self.data = f.readlines()
        self.columns = self.data[0]
        self.data = self.data[1:]
        with open(self.label_path, "r", encoding="UTF8") as f:
            raw_labels = f.readlines()

        self.label = {}
        win_idx = raw_labels[0].split(",").index("_win_0_l")
        matchid_idx = raw_labels[0].split(",").index("_matchId\n")
        for raw_label in raw_labels[1:]:
            raw_label = raw_label.split(",")
            self.label[raw_label[matchid_idx]] = int(raw_label[win_idx])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        matchId = self.data[idx].split(",")[-1]
        data = torch.tensor(
            list(map(float, self.data[idx].split(",")[:-1])),
            dtype=torch.float32,
            requires_grad=True,
        )
        label = torch.tensor(self.label[matchId])

        return data, label


class EasyTabularDataset(Dataset):
    def __init__(self, model_args, data_args, is_train=True):

        if is_train:
            self.data_path = data_args.label_path
        else:
            self.data_path = data_args.test_label_path

        with open(self.data_path, "r", encoding="UTF8") as f:
            raw_labels = f.readlines()

        label_dict = {}
        win_idx = raw_labels[0].split(",").index("_win_0_l")
        matchid_idx = raw_labels[0].split(",").index("_matchId\n")
        for raw_label in raw_labels[1:]:
            raw_label = raw_label.split(",")
            label_dict[raw_label[matchid_idx].strip()] = int(raw_label[win_idx])

        self.data = pd.read_csv(self.data_path)
        self.data.rename(columns={"_matchId": "win"}, inplace=True)
        self.data["win"] = self.data["win"].map(lambda x: label_dict[x])

        self.data.drop(
            [
                "_win_0_l",
                "_win_1_l",
                "_win_2_l",
                "_win_3_l",
                "_win_4_l",
                "_win_5_l",
                "_win_6_l",
                "_win_7_l",
                "_win_8_l",
                "_win_9_l",
            ],
            axis=1,
            inplace=True,
        )

        before_drop_dup = len(self.data)
        self.data.drop_duplicates(self.data.columns.difference(["win"]))
        after_drop_dup = len(self.data)
        print("drop duplicates : ", after_drop_dup - before_drop_dup)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(
            self.data.iloc[idx].values, dtype=torch.float32, requires_grad=True
        )
        label = data[-1]
        data = data[:-1]

        return data, label.long()


class TestTabularDataset(Dataset):
    def __init__(self, model_args, data_args, is_train=True):
        self.data_args = data_args
        self.model_args = model_args
        with open("../data/poker-hand-testing.data", "r", encoding="UTF8") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = torch.tensor(
            list(map(float, self.data[idx].split(","))),
            dtype=torch.float32,
            requires_grad=True,
        )
        x = data[:10]
        label = data[10]

        return x, label.long()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    dataset = EasyTabularDataset(model_args, data_args)
    d = {}
    for x, label in tqdm(dataset):
        if int(label) in d:
            d[int(label)] += 1
        else:
            d[int(label)] = 1
    print(d)
