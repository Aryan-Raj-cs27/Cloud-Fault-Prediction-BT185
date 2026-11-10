| A | B | metric | n_seeds | mean_A | mean_B | mean_diff | wilcoxon_p | ttest_p | A_seed0_95CI |
|---|---|---|---|---|---|---|---|---|---|
| transformer | random_forest | test PR-AUC | 5 | 0.9595 | 0.9447 | 0.0148 | 0.0625 | 0.0146 | [0.949, 0.970] |
| fused | random_forest | test PR-AUC | 5 | 0.9399 | 0.9447 | -0.0048 | 0.1250 | 0.2870 | [0.925, 0.951] |
| random_forest | rule_warn | test PR-AUC | 5 | 0.9447 | 0.1084 | 0.8363 | 0.0625 | 0.0000 | [0.926, 0.952] |
| lstm_ae | random_rate | test PR-AUC | 5 | 0.7818 | 0.0131 | 0.7687 | 0.0625 | 0.0000 | [0.810, 0.854] |
