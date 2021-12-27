from sklearn.metrics import accuracy_score, r2_score

import pandas as pd
import numpy as np

from datasets import load_dataset

from tqdm import tqdm

from xgboost import XGBClassifier

filename = "xgb_model.model"
target = "win"

data_files = {"test": "test.csv"}
dataset = load_dataset(
    "PDJ107/riot-data", data_files=data_files, revision="cgm_20", use_auth_token=True
)

test = dataset["test"].to_pandas()
test = test.drop(["_matchId"], axis=1)

features = [col for col in test.columns if col not in [target]]

X_test = test[features].values[:]
y_test = test[target].values[:]

# 모델 불러오기
new_xgb_model = XGBClassifier()  # 모델 초기화
new_xgb_model.load_model(filename)  # 모델 불러오기

print(X_test.shape)

preds = np.array(new_xgb_model.predict_proba(X_test))
y_true = [0 if x == "lose" else 1 for x in y_test]
y_pred = [0 if x < 0.5 else 1 for x in preds[:, 1]]
test_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
print("test_acc :", test_acc)

r2 = r2_score(y_true, y_pred)
Adj_r2 = 1 - (1 - r2_score(y_true, y_pred)) * (
    (len(X_test) - 1) / (len(X_test) - len(X_test[0]) - 1)
)

print("r2 : ", r2, "Adj_r2 :", Adj_r2)
