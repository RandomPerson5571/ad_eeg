# Feature Selection Results

## Eyes-closed baseline

These values were recorded from the Kaggle `04_feature_selection` notebook output.

### Artifact contracts

| Contract | Value |
|---|---|
| Input path | `/kaggle/temp/ad_eeg/data/features/eyesclosed/baseline/subject_features.parquet` |
| Input rows | 28,882 |
| Subjects | 86 |
| Candidate features | 12 |
| Selected features | 12 of 12 |
| Output rows | 28,882 |

### Mutual-information ranking

| Rank | Source index | Feature | MI score |
|---:|---:|---|---:|
| 1 | 11 | `alpha_wpli` | 1.078419 |
| 2 | 10 | `theta_wpli` | 1.078281 |
| 3 | 7 | `theta_alpha_ratio` | 0.166835 |
| 4 | 9 | `slow_fast_ratio` | 0.163448 |
| 5 | 2 | `rel_alpha` | 0.146901 |
| 6 | 4 | `rel_theta` | 0.087380 |
| 7 | 3 | `rel_beta` | 0.086214 |
| 8 | 5 | `rel_delta` | 0.083979 |
| 9 | 8 | `theta_beta_ratio` | 0.078369 |
| 10 | 6 | `alpha_peak_freq` | 0.047677 |
| 11 | 0 | `lzc_posterior` | 0.027766 |
| 12 | 1 | `mse_posterior` | 0.027493 |

All candidate features passed the configured variance and correlation filters. Because
`top_k` was not set, mutual information ranked the surviving features but did not
remove any of them. The two wPLI connectivity features had the largest univariate
associations with the class label. These scores indicate association, not effect
direction, causality, or non-overlapping predictive information.

Model evaluation must remain subject-grouped. Feature selection used for estimating
generalization performance must be fitted separately within each training fold, as in
the project's nested grouped benchmark.
