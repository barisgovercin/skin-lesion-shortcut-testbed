# Manuscript number check

152 of 152 reported numbers and statements match the values recomputed from results/.

## All checks

| | Where | Manuscript | Recomputed |
|---|---|---|---|
| OK | Table 1 · Vision only · HAM10000 | 0.906±0.008 | 0.9063 ± 0.0079 |
| OK | Table 1 · Vision + metadata · HAM10000 | 0.907±0.008 | 0.9069 ± 0.0077 |
| OK | Table 1 · Vision + synthetic text · HAM10000 | 1.000 | 1.0000 |
| OK | Table 1 · Vision only · ISIC 2020 | 0.982±0.005 | 0.9822 ± 0.0052 |
| OK | Table 1 · Vision + metadata · ISIC 2020 | 0.983±0.005 | 0.9827 ± 0.0055 |
| OK | Table 1 · Vision + synthetic text · ISIC 2020 | 1.000 | 1.0000 |
| OK | Table 1 · Vision only · ISIC-DICM-17K | 0.863±0.005 | 0.8626 ± 0.0052 |
| OK | Table 1 · Vision + metadata · ISIC-DICM-17K | 0.862±0.008 | 0.8625 ± 0.0078 |
| OK | Table 1 · Vision + synthetic text · ISIC-DICM-17K | 1.000 | 1.0000 |
| OK | Dose-response · honest AUROC at p=0.5 · HAM10000 | 0.917 | 0.9174 |
| OK | Dose-response · honest AUROC at p=1.0 · HAM10000 | 0.711 | 0.7114 |
| OK | Dose-response · FFilt-gap at p=1.0 · HAM10000 | 0.289 | 0.2886 |
| OK | Table 3 · SRS at p=1.0 · HAM10000 | 0.998 | 0.9978 |
| OK | Dose-response · Spearman trend (honest) p<0.001 · HAM10000 | <0.001 | 2.855e-06 |
| OK | Dose-response · Spearman trend (srs) p<0.001 · HAM10000 | <0.001 | 1.413e-23 |
| OK | Dose-response · honest AUROC at p=0.5 · ISIC 2020 | 0.985 | 0.9852 |
| OK | Dose-response · honest AUROC at p=1.0 · ISIC 2020 | 0.897 | 0.8968 |
| OK | Dose-response · FFilt-gap at p=1.0 · ISIC 2020 | 0.103 | 0.1032 |
| OK | Table 3 · SRS at p=1.0 · ISIC 2020 | 0.999 | 0.9991 |
| OK | Dose-response · Spearman trend (honest) p<0.001 · ISIC 2020 | <0.001 | 0.0003924 |
| OK | Dose-response · Spearman trend (srs) p<0.001 · ISIC 2020 | <0.001 | 1.413e-23 |
| OK | Dose-response · honest AUROC at p=0.5 · ISIC-DICM-17K | 0.879 | 0.8792 |
| OK | Dose-response · honest AUROC at p=1.0 · ISIC-DICM-17K | 0.679 | 0.6795 |
| OK | Dose-response · FFilt-gap at p=1.0 · ISIC-DICM-17K | 0.321 | 0.3205 |
| OK | Table 3 · SRS at p=1.0 · ISIC-DICM-17K | 0.999 | 0.9988 |
| OK | Dose-response · Spearman trend (honest) p<0.001 · ISIC-DICM-17K | <0.001 | 2.335e-07 |
| OK | Dose-response · Spearman trend (srs) p<0.001 · ISIC-DICM-17K | <0.001 | 1.413e-23 |
| OK | Table 2 · naive honest · HAM10000 | 0.711 | 0.7114 |
| OK | Table 2 · naive SRS · HAM10000 | 0.998 | 0.9978 |
| OK | Table 2 · ffilt_only honest · HAM10000 | 0.918 | 0.9176 |
| OK | Table 2 · ffilt_only SRS · HAM10000 | 0.007 | 0.0067 |
| OK | Table 2 · counterfactual_aug honest · HAM10000 | 0.928 | 0.9276 |
| OK | Table 2 · counterfactual_aug SRS · HAM10000 | 0.004 | 0.0036 |
| OK | Table 2 · consistency honest · HAM10000 | 0.924 | 0.9242 |
| OK | Table 2 · consistency SRS · HAM10000 | 0.002 | 0.0024 |
| OK | Table 2 · adversarial honest · HAM10000 | 0.898 | 0.8978 |
| OK | Table 2 · adversarial SRS · HAM10000 | 0.102 | 0.1016 |
| OK | Table 2 caption · counterfactual_aug has the highest honest AUROC · HAM10000 | 3 | 0.9276 vs others 0.8978, 0.9242, 0.9176, 0.7114 |
| OK | Table 2 caption · counterfactual_aug SRS is near zero (<0.01) · HAM10000 | <0.01 | 0.0036 |
| OK | Table 2 caption · consistency SRS similar or slightly lower than cf · HAM10000 | 3 | 0.0024 vs others 0.0036 |
| OK | Results · method ranking on honest AUROC · HAM10000 | counterfactual_aug>=consistency>ffilt_only>adversarial>naive | counterfactual_aug 0.9276, consistency 0.9242, ffilt_only 0.9176, adversarial 0.8978, naive 0.7114 |
| OK | Table 2 · naive honest · ISIC 2020 | 0.897 | 0.8968 |
| OK | Table 2 · naive SRS · ISIC 2020 | 0.999 | 0.9991 |
| OK | Table 2 · ffilt_only honest · ISIC 2020 | 0.985 | 0.9850 |
| OK | Table 2 · ffilt_only SRS · ISIC 2020 | 0.002 | 0.0021 |
| OK | Table 2 · counterfactual_aug honest · ISIC 2020 | 0.987 | 0.9874 |
| OK | Table 2 · counterfactual_aug SRS · ISIC 2020 | 0.001 | 0.0009767 |
| OK | Table 2 · consistency honest · ISIC 2020 | 0.986 | 0.9864 |
| OK | Table 2 · consistency SRS · ISIC 2020 | 0.001 | 0.0008099 |
| OK | Table 2 · adversarial honest · ISIC 2020 | 0.971 | 0.9714 |
| OK | Table 2 · adversarial SRS · ISIC 2020 | 0.124 | 0.1244 |
| OK | Table 2 caption · counterfactual_aug has the highest honest AUROC · ISIC 2020 | 3 | 0.9874 vs others 0.9714, 0.9864, 0.9850, 0.8968 |
| OK | Table 2 caption · counterfactual_aug SRS is near zero (<0.01) · ISIC 2020 | <0.01 | 0.0009767 |
| OK | Table 2 caption · consistency SRS similar or slightly lower than cf · ISIC 2020 | 3 | 0.0008 vs others 0.0010 |
| OK | Results · method ranking on honest AUROC · ISIC 2020 | counterfactual_aug>=consistency>ffilt_only>adversarial>naive | counterfactual_aug 0.9874, consistency 0.9864, ffilt_only 0.9850, adversarial 0.9714, naive 0.8968 |
| OK | Table 2 · naive honest · ISIC-DICM-17K | 0.679 | 0.6795 |
| OK | Table 2 · naive SRS · ISIC-DICM-17K | 0.999 | 0.9988 |
| OK | Table 2 · ffilt_only honest · ISIC-DICM-17K | 0.880 | 0.8797 |
| OK | Table 2 · ffilt_only SRS · ISIC-DICM-17K | 0.006 | 0.0064 |
| OK | Table 2 · counterfactual_aug honest · ISIC-DICM-17K | 0.889 | 0.8885 |
| OK | Table 2 · counterfactual_aug SRS · ISIC-DICM-17K | 0.003 | 0.0033 |
| OK | Table 2 · consistency honest · ISIC-DICM-17K | 0.887 | 0.8869 |
| OK | Table 2 · consistency SRS · ISIC-DICM-17K | 0.003 | 0.0033 |
| OK | Table 2 · adversarial honest · ISIC-DICM-17K | 0.784 | 0.7842 |
| OK | Table 2 · adversarial SRS · ISIC-DICM-17K | 0.679 | 0.6795 |
| OK | Table 2 caption · counterfactual_aug has the highest honest AUROC · ISIC-DICM-17K | 3 | 0.8885 vs others 0.7842, 0.8869, 0.8797, 0.6795 |
| OK | Table 2 caption · counterfactual_aug SRS is near zero (<0.01) · ISIC-DICM-17K | <0.01 | 0.0033 |
| OK | Table 2 caption · consistency SRS similar or slightly lower than cf · ISIC-DICM-17K | 3 | 0.0033 vs others 0.0033 |
| OK | Results · method ranking on honest AUROC · ISIC-DICM-17K | counterfactual_aug>=consistency>ffilt_only>adversarial>naive | counterfactual_aug 0.8885, consistency 0.8869, ffilt_only 0.8797, adversarial 0.7842, naive 0.6795 |
| OK | Results · naive Flip Rate · HAM10000 | 1.000 | 1.0000 |
| OK | Results · naive Flip Rate · ISIC 2020 | 1.000 | 1.0000 |
| OK | Results · naive Flip Rate · ISIC-DICM-17K | 1.000 | 1.0000 |
| OK | Results · counterfactual_aug Flip Rate · HAM10000 | 0.004 | 0.0040 |
| OK | Results · counterfactual_aug Flip Rate · ISIC 2020 | 0.001 | 0.0012 |
| OK | Results · counterfactual_aug Flip Rate · ISIC-DICM-17K | 0.004 | 0.0036 |
| OK | Results · adversarial Flip Rate · ISIC-DICM-17K | 0.876 | 0.8763 |
| OK | Results · adversarial SRS ~0.68 · ISIC-DICM-17K | 0.68 | 0.6795 |
| OK | Results · cf vs filtering, p · HAM10000 | 0.006 | 0.0058 |
| OK | Results · cf vs filtering, mean gain · HAM10000 | +0.010 | 0.0100 |
| OK | Results · cf vs filtering, 95% CI low · HAM10000 | +0.005 | 0.0048 |
| OK | Results · cf vs filtering, 95% CI high · HAM10000 | +0.015 | 0.0152 |
| OK | Results · cf vs filtering, Cohen's d · HAM10000 | 2.4 | 2.3971 |
| OK | Contribution 4 · cf beats filtering on SRS (p<0.05) · HAM10000 | p<0.05 | 0.0088 |
| OK | Results · cf vs filtering, p · ISIC 2020 | 0.001 | 0.0008133 |
| OK | Results · cf vs filtering, mean gain · ISIC 2020 | +0.002 | 0.0024 |
| OK | Results · cf vs filtering, 95% CI low · ISIC 2020 | +0.002 | 0.0017 |
| OK | Results · cf vs filtering, 95% CI high · ISIC 2020 | +0.003 | 0.0031 |
| OK | Results · cf vs filtering, Cohen's d · ISIC 2020 | 4.1 | 4.0637 |
| OK | Contribution 4 · cf beats filtering on SRS (p<0.05) · ISIC 2020 | p<0.05 | 0.0037 |
| OK | Results · cf vs filtering, p · ISIC-DICM-17K | 0.037 | 0.0369 |
| OK | Results · cf vs filtering, mean gain · ISIC-DICM-17K | +0.009 | 0.0088 |
| OK | Results · cf vs filtering, 95% CI low · ISIC-DICM-17K | +0.001 | 0.0008731 |
| OK | Results · cf vs filtering, 95% CI high · ISIC-DICM-17K | +0.017 | 0.0168 |
| OK | Results · cf vs filtering, Cohen's d · ISIC-DICM-17K | 1.4 | 1.3780 |
| OK | Contribution 4 · cf beats filtering on SRS (p<0.05) · ISIC-DICM-17K | p<0.05 | 0.0121 |
| OK | Results · cf vs naive p · HAM10000 | <0.001 | 0.000785 |
| OK | Results · cf vs naive, mean gain · HAM10000 | +0.216 | 0.2162 |
| OK | Results · cf vs naive, d>2 · HAM10000 | >2 | 4.1013 |
| OK | Results · cf vs naive p · ISIC 2020 | 0.007 | 0.0067 |
| OK | Results · cf vs naive, mean gain · ISIC 2020 | +0.091 | 0.0907 |
| OK | Results · cf vs naive, d>2 · ISIC 2020 | >2 | 2.3046 |
| OK | Results · cf vs naive p · ISIC-DICM-17K | <0.001 | 0.0004606 |
| OK | Results · cf vs naive, mean gain · ISIC-DICM-17K | +0.209 | 0.2090 |
| OK | Results · cf vs naive, d>2 · ISIC-DICM-17K | >2 | 4.7077 |
| OK | Results · consistency indistinguishable from cf · ISIC 2020 | p>=0.05 | 0.1758 |
| OK | Results · cf vs consistency, p · ISIC 2020 | 0.18 | 0.1758 |
| OK | Results · consistency indistinguishable from cf · ISIC-DICM-17K | p>=0.05 | 0.4287 |
| OK | Results · cf vs consistency, p · ISIC-DICM-17K | 0.43 | 0.4287 |
| OK | Results · cf vs consistency, p · HAM10000 | 0.04 | 0.0425 |
| OK | Results · cf vs consistency, mean difference · HAM10000 | 0.003 | 0.0034 |
| OK | Control · budget-matched filtering honest AUROC | 0.926 | 0.9260 |
| OK | Control · counterfactual_aug honest AUROC | 0.928 | 0.9276 |
| OK | Control · cf minus budget-matched filtering, honest AUROC | +0.002 | 0.0015 |
| OK | Control · 95% CI low | -0.002 | -0.0020 |
| OK | Control · 95% CI high | +0.005 | 0.0051 |
| OK | Control · p | 0.30 | 0.2968 |
| OK | Control · cf matches budget-matched filtering (not significant) | p>=0.05 | 0.2968 |
| OK | Control · counterfactual_aug SRS | 0.004 | 0.0036 |
| OK | Control · budget-matched filtering SRS | 0.006 | 0.0055 |
| OK | Control · cf keeps a lower SRS | <0 | -0.0020 |
| OK | Control · lower SRS, p | 0.015 | 0.0147 |
| OK | Control · lower SRS is significant | p<0.05 | 0.0147 |
| OK | LLM · honest AUROC holds near 0.91 before collapsing | 0.90-0.92 | 0.9169, 0.9162, 0.9140, 0.9084, 0.9033 |
| OK | LLM · Spearman rho for SRS vs p | 0.99 | 0.9866 |
| OK | LLM · Spearman p<0.001 for SRS vs p | <0.001 | 1.413e-23 |
| OK | LLM · counterfactual_aug honest AUROC | 0.928 | 0.9282 |
| OK | LLM · counterfactual_aug SRS | 0.006 | 0.0062 |
| OK | LLM · naive honest AUROC | 0.656 | 0.6559 |
| OK | LLM · cf vs naive p<0.001 | <0.001 | 0.000159 |
| OK | LLM · cf vs filtering p | 0.003 | 0.0027 |
| OK | LLM · cf vs adversarial p | 0.019 | 0.0190 |
| OK | LLM · cf vs consistency p | 0.048 | 0.0477 |
| OK | Table 4 · model NB at 0.1 · HAM10000 | 0.153 | 0.1527 |
| OK | Table 4 · refer-all NB at 0.1 · HAM10000 | 0.106 | 0.1060 |
| OK | Table 4 · model NB at 0.3 · HAM10000 | 0.101 | 0.1007 |
| OK | Table 4 · refer-all NB at 0.3 · HAM10000 | -0.149 | -0.1494 |
| OK | DCA · margin over refer-all excludes zero at every threshold · HAM10000 | >0 | 0.0174 |
| OK | DCA · model net benefit positive at every threshold · HAM10000 | >0 | 0.1007 |
| OK | Table 4 · model NB at 0.1 · ISIC 2020 | 0.118 | 0.1178 |
| OK | Table 4 · refer-all NB at 0.1 · ISIC 2020 | 0.036 | 0.0362 |
| OK | Table 4 · model NB at 0.3 · ISIC 2020 | 0.110 | 0.1099 |
| OK | Table 4 · refer-all NB at 0.3 · ISIC 2020 | -0.239 | -0.2392 |
| OK | DCA · margin over refer-all excludes zero at every threshold · ISIC 2020 | >0 | 0.0334 |
| OK | DCA · model net benefit positive at every threshold · ISIC 2020 | >0 | 0.1099 |
| OK | Table 4 · model NB at 0.1 · ISIC-DICM-17K | 0.463 | 0.4632 |
| OK | Table 4 · refer-all NB at 0.1 · ISIC-DICM-17K | 0.453 | 0.4528 |
| OK | Table 4 · model NB at 0.3 · ISIC-DICM-17K | 0.379 | 0.3785 |
| OK | Table 4 · refer-all NB at 0.3 · ISIC-DICM-17K | 0.296 | 0.2964 |
| OK | DCA · margin over refer-all excludes zero at every threshold · ISIC-DICM-17K | >0 | 0.0021 |
| OK | DCA · model net benefit positive at every threshold · ISIC-DICM-17K | >0 | 0.3785 |
| OK | DCA · refer-all harmful at 0.30 · HAM10000 | <0 | -0.1494 |
| OK | DCA · refer-all harmful at 0.30 · ISIC 2020 | <0 | -0.2392 |
