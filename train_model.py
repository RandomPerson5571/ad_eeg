from util import load_features

import pandas as pd

df = pd.read_parquet("parquet_files/all_features.parquet")

# print(df.columns.tolist())
# print(df.head())
# 'rel_alpha', 'rel_beta', 'rel_theta', 'rel_delta', 'alpha_peak_freq', 'theta_alpha_ratio', 'theta_beta_ratio', 'slow_fast_ratio', 'lzc', 'mse_mean', 'participant_id', 'dataset_id', 'label']

features, labels = load_features()

print(features)
print(labels)