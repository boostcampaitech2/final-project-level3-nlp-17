from torch.utils.data import Dataset
import torch

from tqdm import tqdm

from transformers import HfArgumentParser
from arguments import DataArguments, ModelArguments
import pandas as pd

class TabularDatasetFromHuggingface(Dataset):
    def __init__(self, dataset):
        
        self.data = []
        self.label = []

        self.columns = dataset[0].keys()

        for data in tqdm(dataset):
            data = list(data.values())
            self.data.append(list(map(float, data[:-2])))
            self.label.append(int(data[-1]))

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32, requires_grad=True)
        label = torch.tensor(self.label[idx])

        return data, label

class TabularDataset(Dataset):
    def __init__(self, model_args, data_args, is_train=True):

        if is_train:
            self.data_path = data_args.data_path
            self.label_path = data_args.label_path
        else:
            self.data_path = data_args.test_data_path
            self.label_path = data_args.test_label_path

        with open(self.data_path, 'r', encoding='UTF8') as f:
            self.data = f.readlines()
        self.columns = self.data[0]
        self.data = self.data[1:]
        with open(self.label_path, 'r', encoding='UTF8') as f:
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


class EasyTabularDataset(Dataset):
    def __init__(self, model_args, data_args, is_train=True):

        if is_train:
            self.data_path = data_args.label_path
        else:
            self.data_path = data_args.test_label_path

        with open(self.data_path, 'r', encoding='UTF8') as f:
            raw_labels = f.readlines()

        label_dict={}
        win_idx = raw_labels[0].split(',').index('_win_0_l')
        matchid_idx = raw_labels[0].split(',').index('_matchId\n')
        for raw_label in raw_labels[1:]:
            raw_label = raw_label.split(',')
            label_dict[raw_label[matchid_idx].strip()] = int(raw_label[win_idx])

        self.data = pd.read_csv(self.data_path)
        self.data.rename(columns = {'_matchId': 'win'}, inplace = True)
        self.data['win'] = self.data['win'].map(lambda x: label_dict[x])

        self.data.drop(['_win_0_l', '_win_1_l', '_win_2_l', '_win_3_l', '_win_4_l', '_win_5_l', '_win_6_l', '_win_7_l', '_win_8_l', '_win_9_l'], axis=1, inplace=True)
        
        before_drop_dup = len(self.data)
        self.data.drop_duplicates(self.data.columns.difference(['win']))
        after_drop_dup = len(self.data)
        print('drop duplicates : ', after_drop_dup - before_drop_dup)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32, requires_grad=True)
        label = data[-1]
        data = data[:-1]

        return data, label.long()


class InnerEvalDataset(Dataset):
    '''
        현실 게임에서는 거의 대부분 챌린저가 브론즈를 이긴다. 따라서 모델의 성능을 평가할 때 챌린저와 브론즈간 가상의 매치(경기)를 잡고 inference를 했을 때 
        모델은 챌린저가 이길 확률이 (95%이상으로) 매우 높다고 한다면 실제 사실과 모델의 예측이 같기 때문에 모델이 데이터를 잘 학습했다고 평가할 수 있다.
        이를 확인하기위해 학습 데이터셋을 재조합하여 가상의 매치 데이터를 반환한다.

        a : 특정 티어 ex) 챌린저
        b : a가 아닌 특정 티어 ex) 브론즈
        left(l) : 특정 매치의 20경기 평균 데이터 (==학습 데이터)에서 왼쪽에 있는 5명의 데이터
        right(r) : 특정 매치의 20경기 평균 데이터 (==학습 데이터)에서 오른쪽에 있는 5명의 데이터

        a_left_a_right (alar) (== "챌린저 왼쪽 챌린저 오른쪽") : a("챌린저") 학습 데이터의 한 행(1*270)에서 0~135 열과 a("챌린저") 학습 데이터의 한 행 (1*270)에서 135~270 열을 concat한 데이터
        a_left_b_right (albr) (== "챌린저 왼쪽 브론즈 오른쪽")
        b_left_a_right (blar) (== "챌린저 오른쪽 브론즈 왼쪽")
        b_left_b_right (blbr) (== "챌린저 오른쪽 브론즈 오른쪽")
        
               ar     br
            ---------------
        al  | alar | albr |
            ---------------
        bl  | blar | blbr |

        a가 항상 b를 이기는 경우

              ar    br
            -------------
        al  | 50% | 99% |
            -------------
        bl  | 00% | 50% |
        
        가 나올것이다.

    '''
    def __init__(self, a_team_data_path, a_team_label_path, b_team_data_path, b_team_label_path):

        with open(a_team_data_path, 'r', encoding='UTF8') as f:
            self.a_data = f.readlines()
        self.a_bumns = self.a_data[0]
        self.a_data = self.a_data[1:]
        with open(a_team_label_path, 'r', encoding='UTF8') as f:
            raw_a_labels = f.readlines()
        
        self.a_label={}
        win_idx = raw_a_labels[0].split(',').index('_win_0_l')
        matchid_idx = raw_a_labels[0].split(',').index('_matchId\n')
        for raw_label in raw_a_labels[1:]:
            raw_label = raw_label.split(',')
            self.a_label[raw_label[matchid_idx]] = int(raw_label[win_idx])

        with open(b_team_data_path, 'r', encoding='UTF8') as f:
            self.b_data = f.readlines()
        self.b_bumns = self.b_data[0]
        self.b_data = self.b_data[1:]
        with open(b_team_label_path, 'r', encoding='UTF8') as f:
            raw_b_labels = f.readlines()
        
        self.b_label={}
        win_idx = raw_b_labels[0].split(',').index('_win_0_l')
        matchid_idx = raw_b_labels[0].split(',').index('_matchId\n')
        for raw_label in raw_b_labels[1:]:
            raw_label = raw_label.split(',')
            self.b_label[raw_label[matchid_idx]] = int(raw_label[win_idx])


    def __len__(self):
        return min(len(self.a_data), len(self.b_data))

    def __getitem__(self, idx):

        a_matchId = self.a_data[idx].split(',')[-1]
        a_data = torch.tensor(list(map(float, self.a_data[idx].split(',')[:-1])), dtype=torch.float32, requires_grad=True)
        a_label = torch.tensor(self.a_label[a_matchId])

        b_matchId = self.b_data[idx].split(',')[-1]
        b_data = torch.tensor(list(map(float, self.b_data[idx].split(',')[:-1])), dtype=torch.float32, requires_grad=True)
        b_label = torch.tensor(self.b_label[b_matchId])

        mid = min(len(a_data), len(b_data))//2

        alar_data = torch.cat([a_data[:mid], a_data[mid:]]) 
        albr_data = torch.cat([a_data[:mid], b_data[mid:]]) 
        blar_data = torch.cat([b_data[:mid], a_data[mid:]]) 
        blbr_data = torch.cat([b_data[:mid], b_data[mid:]]) 

        return (alar_data, albr_data, blar_data, blbr_data)

if __name__=='__main__':
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