| A | B | metric | n_seeds | mean_A | mean_B | mean_diff | wilcoxon_p | ttest_p | A_seed0_95CI |
|---|---|---|---|---|---|---|---|---|---|
| transformer | logreg | test PR-AUC | 5 | 0.8309 | 0.8258 | 0.0050 | 0.3125 | 0.3010 | [0.797, 0.844] |
| fused | transformer | test PR-AUC | 5 | 0.8345 | 0.8309 | 0.0036 | 0.4375 | 0.2171 | [0.805, 0.851] |
| transformer | rule_warn | test PR-AUC | 5 | 0.8309 | 0.0153 | 0.8156 | 0.0625 | 0.0000 | [0.797, 0.844] |
| iforest | random_rate | test PR-AUC | 5 | 0.6396 | 0.0131 | 0.6265 | 0.0625 | 0.0000 | [0.607, 0.666] |
