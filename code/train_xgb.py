from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

from datasets import load_dataset

from tqdm import tqdm

from xgboost import XGBClassifier

filename = "xgb_model.model"
target = "win"
max_epochs = 50


data_files = {"train": "train.csv", "validation": "validation.csv"}
dataset = load_dataset("PDJ107/riot-data", data_files=data_files, revision="cgm_20")

train = dataset["train"].to_pandas()
train = train.drop(["_matchId"], axis=1)
train["Set"] = "train"
valid = dataset["validation"].to_pandas()
valid = valid.drop(["_matchId"], axis=1)
valid["Set"] = "valid"

train = pd.concat([train, valid], ignore_index=True)

before_len = len(train)
train["zero_num"] = train.isin([0]).sum(axis=1)
train = train.drop(train[train["zero_num"] >= 315].index)
train = train.drop(["zero_num"], axis=1)
after_len = len(train)
print("before_len :", before_len)
print("remove :", after_len - before_len)
train = train.reset_index()

train_indices = train[train.Set == "train"].index.sort_values()
valid_indices = train[train.Set == "valid"].index.sort_values()

nunique = train.nunique()
types = train.dtypes

categorical_columns = []
categorical_dims = {}
for col in tqdm(train.columns):
    if types[col] == "object" or nunique[col] < 200:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)


train.loc[train[target] == 0, target] = "lose"
train.loc[train[target] == 1, target] = "win"

unused_feat = ["Set"]

features = [col for col in train.columns if col not in unused_feat + [target]]

cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [
    categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns
]


X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

clf_xgb = XGBClassifier(
    max_depth=8,
    learning_rate=0.1,
    n_estimators=1000,
    verbosity=0,
    silent=None,
    objective="binary:logistic",
    booster="gbtree",
    n_jobs=-1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,
)

clf_xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=200,
    verbose=10,
)

preds = np.array(clf_xgb.predict_proba(X_valid))
valid_acc = accuracy_score(
    y_true=[0 if x == "lose" else 1 for x in y_valid],
    y_pred=[0 if x < 0.5 else 1 for x in preds[:, 1]],
)
print(valid_acc)

# 모델 저장
clf_xgb.save_model(filename)
