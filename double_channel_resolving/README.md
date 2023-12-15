## Problem: some markers have been recorded using two or even three staining agents (e.g. "FITC", "PE", or "AF88")

- Resolve these redundancies by running double_channel_resolving.py:
    - Calculates the effect of the pre-processing as the MSE of the image before and after
    - Small MSE: image did not "need" much pre-processing -> is assumed to have good quality
    - MSE is calculated for all samples with redundant markers and summed up

- double_channel_resolving_{condition}.txt contains global results
- effect_of_preprocessing.csv contains sample-wise marker-wise MSEs, gloabl results are calculated by summing across samples
