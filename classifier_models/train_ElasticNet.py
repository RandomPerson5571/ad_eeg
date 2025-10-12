import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from util import load_features

features, labels = load_features()

# Labels
# A - Alzheimers
# C - Control (Healthy)
# F - Frontroltemporal Dementia

df = pd.read_parquet("parquet_files/all_features.parquet")

groups_array = df.iloc[0].to_numpy()

clf=LogisticRegression()
gkf=GroupKFold(5)
pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
param_grid={'clf__C':[0.1, 0.5, 0.7, 1, 3, 5, 7]}
gscv=GridSearchCV(pipe, param_grid, cv=gkf, n_jobs=12)
gscv.fit(features, labels, groups=groups_array)

print(gscv.best_score_)