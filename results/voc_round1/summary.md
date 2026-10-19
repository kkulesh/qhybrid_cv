# Round 1 results

## voc

| Model | qubits / dim | Trainable params | Accuracy | Macro-F1 | ms / image (head only) |
|---|---|---|---|---|---|
| classical: linear_bottleneck | 4 | 5224 | 0.731 | 0.730 ± 0.011 | 0.002 |
| classical: linear_bottleneck | 6 | 7826 | 0.802 | 0.801 ± 0.003 | 0.001 |
| classical: linear_bottleneck | 8 | 10428 | 0.815 | 0.815 ± 0.003 | 0.002 |
| classical: linear_bottleneck | 15 | 19535 | 0.850 | 0.850 ± 0.006 | 0.001 |
| classical: linear_bottleneck | 63 | 81983 | 0.780 | 0.780 ± 0.041 | 0.003 |
| classical: linear_bottleneck | 255 | 331775 | 0.806 | 0.809 ± 0.012 | 0.008 |
| classical: logreg_1280 | 1280 | 25620 | 0.872 | 0.872 | 0.004 |
| classical: mlp | 4 | 5244 | 0.171 | 0.064 ± 0.011 | 0.002 |
| classical: mlp | 6 | 7868 | 0.355 | 0.329 ± 0.459 | 0.002 |
| classical: mlp | 8 | 10500 | 0.598 | 0.563 ± 0.049 | 0.001 |
| hybrid: amplitude (γ=2; at γ=1: F1 0.765) | 4 | 19347 (VQC 24) | 0.783 | 0.783 ± 0.008 | 0.029 |
| hybrid: angle (γ=0.1; at γ=1: F1 0.092) | 4 | 5256 (VQC 24) | 0.235 | 0.133 ± 0.083 | 0.032 |
| hybrid: zz (γ=0.02; at γ=1: F1 0.021) | 4 | 5256 (VQC 24) | 0.447 | 0.357 ± 0.022 | 0.075 |
| hybrid: amplitude (γ=0.25; at γ=1: F1 0.830) | 6 | 80891 (VQC 36) | 0.841 | 0.841 ± 0.006 | 0.043 |
| hybrid: angle (γ=0.02; at γ=1: F1 0.144) | 6 | 7874 (VQC 36) | 0.535 | 0.482 ± 0.023 | 0.063 |
| hybrid: zz (γ=0.02; at γ=1: F1 0.005) | 6 | 7874 (VQC 36) | 0.453 | 0.375 ± 0.028 | 0.200 |
| hybrid: amplitude (γ=1; at γ=1: F1 0.853) | 8 | 326899 (VQC 48) | 0.853 | 0.853 ± 0.003 | 0.088 |
| hybrid: angle (γ=0.05; at γ=1: F1 0.114) | 8 | 10492 (VQC 48) | 0.458 | 0.373 ± 0.040 | 0.147 |
| hybrid: zz (γ=0.02; at γ=1: F1 0.018) | 8 | 10492 (VQC 48) | 0.628 | 0.599 ± 0.022 | 0.715 |

## Method A: correlation of a-priori metrics with trained test F1

Spearman ρ over all (encoding, γ) configurations within each dataset × qubit count, then averaged.

| Metric | mean ρ | min ρ | max ρ |
|---|---|---|---|
| kta | 0.76 | 0.71 | 0.80 |
| hs | -0.14 | -0.20 | -0.10 |
| probe | 0.36 | 0.09 | 0.66 |
| combo | 0.43 | 0.35 | 0.54 |
| var | -0.16 | -0.26 | -0.03 |

## Method A: choosing encoding and γ without training

Oracle = best test F1 over all trained configs. Regret = oracle F1 − F1 of the chosen config. Short training = pick config by validation F1 after 1 or 2 epochs (needs every config trained that long).

| Selector | top-1 encoding hit rate | mean regret (F1) | max regret | cost (s, all configs) |
|---|---|---|---|---|
| kta | 100% | 0.002 | 0.005 | 16 |
| hs | 0% | 0.678 | 0.697 | 16 |
| probe | 67% | 0.220 | 0.646 | 16 |
| combo | 67% | 0.251 | 0.742 | 16 |
| short training (1 epoch) | 100% | 0.002 | 0.005 | 57 |
| short training (2 epochs) | 100% | 0.005 | 0.014 | 115 |
| always angle, γ=1 | 0% | 0.710 | 0.742 | 0 |
| full training (val F1) | 100% | 0.001 | 0.004 | 574 |

Per case (dataset, qubits): oracle vs choice of `kta` and of 1-epoch training

| dataset | n | oracle | kta picks | 1-epoch picks |
|---|---|---|---|---|
| voc | 4 | amplitude γ=2 (F1 0.783) | amplitude γ=2 (−0.000) | amplitude γ=0.5 (−0.002) |
| voc | 6 | amplitude γ=0.25 (F1 0.841) | amplitude γ=2 (−0.005) | amplitude γ=0.25 (−0.000) |
| voc | 8 | amplitude γ=0.1 (F1 0.856) | amplitude γ=2 (−0.002) | amplitude γ=0.25 (−0.005) |

## Method A: choosing γ for a fixed encoding

| Selector | mean regret (F1) | max regret |
|---|---|---|
| kta | 0.168 | 0.479 |
| hs | 0.128 | 0.346 |
| probe | 0.114 | 0.324 |
| combo | 0.167 | 0.479 |
| fixed γ=1 | 0.217 | 0.580 |