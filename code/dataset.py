from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch

from tqdm import tqdm

from transformers import HfArgumentParser
from arguments import DataArguments, ModelArguments
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import numpy as np


def create_dataloaders(
    train_dataset,
    val_dataset,
    weights=1,
    batch_size=1024,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
):
    need_shuffle, sampler = create_sampler(
        weights, train_dataset.prepare_target(train_dataset.label)
    )

    print(need_shuffle, sampler)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=need_shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dataloader, val_dataloader


def construct_loaders(train_dataset, val_dataset, weights, device):

    train_dataloader, valid_dataloader = create_dataloaders(
        train_dataset, val_dataset, weights=weights, pin_memory=device.type != "cpu"
    )

    return train_dataloader, valid_dataloader


def create_sampler(weights, y_train):

    if isinstance(weights, int):
        if weights == 0:
            need_shuffle = True
            sampler = None
        elif weights == 1:
            need_shuffle = False
            class_sample_count = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
            )

            weights = 1.0 / class_sample_count

            samples_weight = np.array([weights[t] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            raise ValueError("Weights should be either 0, 1, dictionnary or list.")
    elif isinstance(weights, dict):
        # custom weights per class
        need_shuffle = False
        samples_weight = np.array([weights[t] for t in y_train])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        # custom weights
        if len(weights) != len(y_train):
            raise ValueError("Custom weights should match number of train samples.")
        need_shuffle = False
        samples_weight = np.array(weights)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return need_shuffle, sampler


class TabularDatasetFromHuggingface(Dataset):
    def __init__(self, dataset, is_train=True):

        dataset = dataset.to_pandas()
        self.dataset = dataset.drop(["_matchId"], axis=1)
        self.columns = self.dataset.columns

        if is_train:
            before_len = len(dataset)
            dataset["zero_num"] = dataset.isin([0]).sum(axis=1)
            dataset = dataset.drop(dataset[dataset["zero_num"] >= 315].index)
            # dataset = dataset.sort_values(["zero_num"])
            dataset = dataset.drop(["zero_num"], axis=1)
            after_len = len(dataset)
            print("before_len :", before_len)
            print("remove :", after_len - before_len)

        target = "win"

        self.target_mapper = {"win": 1, "lose": 0}

        nunique = self.dataset.nunique()
        types = self.dataset.dtypes

        categorical_columns = []
        categorical_dims = {}
        for col in tqdm(self.dataset.columns):
            if types[col] == "object" or nunique[col] < 200:
                print(col, self.dataset[col].nunique())
                l_enc = LabelEncoder()
                self.dataset[col] = self.dataset[col].fillna("VV_likely")
                self.dataset[col] = l_enc.fit_transform(self.dataset[col].values)
                categorical_columns.append(col)
                categorical_dims[col] = len(l_enc.classes_)
            else:
                self.dataset.fillna(self.dataset.loc[:, col].mean(), inplace=True)

        # check that pipeline accepts strings
        self.dataset.loc[self.dataset[target] == 0, target] = "lose"
        self.dataset.loc[self.dataset[target] == 1, target] = "win"

        self.features = [col for col in self.dataset.columns if col not in [target]]

        self.cat_idxs = [
            i for i, f in enumerate(self.features) if f in categorical_columns
        ]

        self.cat_dims = [
            categorical_dims[f]
            for i, f in enumerate(self.features)
            if f in categorical_columns
        ]

        self.data = self.dataset[self.features].values[:].astype(np.float32)
        self.label = self.dataset[target].values[:]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.prepare_target(self.label[idx])
        return data, label

    def prepare_target(self, y):
        return np.vectorize(self.target_mapper.get)(y)


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


class InnerEvalDataset(Dataset):
    """
    ?????? ??????????????? ?????? ????????? ???????????? ???????????? ?????????. ????????? ????????? ????????? ????????? ??? ???????????? ???????????? ????????? ??????(??????)??? ?????? inference??? ?????? ???
    ????????? ???????????? ?????? ????????? (95%????????????) ?????? ????????? ????????? ?????? ????????? ????????? ????????? ?????? ????????? ????????? ???????????? ??? ??????????????? ????????? ??? ??????.
    ?????? ?????????????????? ?????? ??????????????? ??????????????? ????????? ?????? ???????????? ????????????.

    a : ?????? ?????? ex) ?????????
    b : a??? ?????? ?????? ?????? ex) ?????????
    left(l) : ?????? ????????? 20?????? ?????? ????????? (==?????? ?????????)?????? ????????? ?????? 5?????? ?????????
    right(r) : ?????? ????????? 20?????? ?????? ????????? (==?????? ?????????)?????? ???????????? ?????? 5?????? ?????????

    a_left_a_right (alar) (== "????????? ?????? ????????? ?????????") : a("?????????") ?????? ???????????? ??? ???(1*270)?????? 0~135 ?????? a("?????????") ?????? ???????????? ??? ??? (1*270)?????? 135~270 ?????? concat??? ?????????
    a_left_b_right (albr) (== "????????? ?????? ????????? ?????????")
    b_left_a_right (blar) (== "????????? ????????? ????????? ??????")
    b_left_b_right (blbr) (== "????????? ????????? ????????? ?????????")

           ar     br
        ---------------
    al  | alar | albr |
        ---------------
    bl  | blar | blbr |

    a??? ?????? b??? ????????? ??????

          ar    br
        -------------
    al  | 50% | 99% |
        -------------
    bl  | 00% | 50% |

    ??? ???????????????.

    """

    def __init__(
        self, a_team_data_path, a_team_label_path, b_team_data_path, b_team_label_path
    ):

        with open(a_team_data_path, "r", encoding="UTF8") as f:
            self.a_data = f.readlines()
        self.a_bumns = self.a_data[0]
        self.a_data = self.a_data[1:]
        with open(a_team_label_path, "r", encoding="UTF8") as f:
            raw_a_labels = f.readlines()

        self.a_label = {}
        win_idx = raw_a_labels[0].split(",").index("_win_0_l")
        matchid_idx = raw_a_labels[0].split(",").index("_matchId\n")
        for raw_label in raw_a_labels[1:]:
            raw_label = raw_label.split(",")
            self.a_label[raw_label[matchid_idx]] = int(raw_label[win_idx])

        with open(b_team_data_path, "r", encoding="UTF8") as f:
            self.b_data = f.readlines()
        self.b_bumns = self.b_data[0]
        self.b_data = self.b_data[1:]
        with open(b_team_label_path, "r", encoding="UTF8") as f:
            raw_b_labels = f.readlines()

        self.b_label = {}
        win_idx = raw_b_labels[0].split(",").index("_win_0_l")
        matchid_idx = raw_b_labels[0].split(",").index("_matchId\n")
        for raw_label in raw_b_labels[1:]:
            raw_label = raw_label.split(",")
            self.b_label[raw_label[matchid_idx]] = int(raw_label[win_idx])

    def __len__(self):
        return min(len(self.a_data), len(self.b_data))

    def __getitem__(self, idx):

        a_matchId = self.a_data[idx].split(",")[-1]
        a_data = torch.tensor(
            list(map(float, self.a_data[idx].split(",")[:-1])),
            dtype=torch.float32,
            requires_grad=True,
        )
        a_label = torch.tensor(self.a_label[a_matchId])

        b_matchId = self.b_data[idx].split(",")[-1]
        b_data = torch.tensor(
            list(map(float, self.b_data[idx].split(",")[:-1])),
            dtype=torch.float32,
            requires_grad=True,
        )
        b_label = torch.tensor(self.b_label[b_matchId])

        mid = min(len(a_data), len(b_data)) // 2

        alar_data = torch.cat([a_data[:mid], a_data[mid:]])
        albr_data = torch.cat([a_data[:mid], b_data[mid:]])
        blar_data = torch.cat([b_data[:mid], a_data[mid:]])
        blbr_data = torch.cat([b_data[:mid], b_data[mid:]])

        return (alar_data, albr_data, blar_data, blbr_data)


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
