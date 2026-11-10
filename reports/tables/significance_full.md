| A | B | metric | n_seeds | mean_A | mean_B | mean_diff | wilcoxon_p | ttest_p | A_seed0_95CI |
|---|---|---|---|---|---|---|---|---|---|
| transformer | xgboost | test PR-AUC | 5 | 0.9999 | 1.0000 | -0.0001 | 0.1875 | 0.1758 | [1.000, 1.000] |
| fused | xgboost | test PR-AUC | 5 | 1.0000 | 1.0000 | 0.0000 | 0.6250 | 0.7103 | [1.000, 1.000] |
| xgboost | rule_warn | test PR-AUC | 5 | 1.0000 | 0.1758 | 0.8241 | 0.0625 | 0.0000 | [1.000, 1.000] |
| lstm_ae | random_rate | test PR-AUC | 5 | 0.8493 | 0.0131 | 0.8362 | 0.0625 | 0.0001 | [0.706, 0.754] |
