# Round 1 results

## cifar10

| Model | qubits / dim | Trainable params | Accuracy | Macro-F1 | ms / image (head only) |
|---|---|---|---|---|---|
| classical: linear_bottleneck | 4 | 5174 | 0.739 | 0.736 ± 0.081 | 0.001 |
| classical: linear_bottleneck | 6 | 7756 | 0.828 | 0.828 ± 0.018 | 0.001 |
| classical: linear_bottleneck | 8 | 10338 | 0.826 | 0.827 ± 0.007 | 0.001 |
| classical: linear_bottleneck | 15 | 19375 | 0.833 | 0.832 ± 0.003 | 0.002 |
| classical: linear_bottleneck | 63 | 81343 | 0.814 | 0.813 ± 0.010 | 0.003 |
| classical: linear_bottleneck | 255 | 329215 | 0.789 | 0.788 ± 0.003 | 0.008 |
| classical: logreg_1280 | 1280 | 12810 | 0.844 | 0.843 | 0.003 |
| classical: mlp | 4 | 5194 | 0.552 | 0.504 ± 0.084 | 0.002 |
| classical: mlp | 6 | 7798 | 0.500 | 0.429 ± 0.285 | 0.001 |
| classical: mlp | 8 | 10410 | 0.703 | 0.695 ± 0.055 | 0.001 |
| hybrid: amplitude (γ=0.25; at γ=1: F1 0.831) | 4 | 19289 (VQC 24) | 0.844 | 0.843 ± 0.002 | 0.025 |
| hybrid: angle (γ=1; at γ=1: F1 0.141) | 4 | 5198 (VQC 24) | 0.258 | 0.141 ± 0.104 | 0.030 |
| hybrid: zz (γ=0.25; at γ=1: F1 0.052) | 4 | 5198 (VQC 24) | 0.189 | 0.082 ± 0.011 | 0.070 |
| hybrid: amplitude (γ=0.1; at γ=1: F1 0.850) | 6 | 80809 (VQC 36) | 0.857 | 0.856 ± 0.004 | 0.052 |
| hybrid: angle (γ=0.25; at γ=1: F1 0.045) | 6 | 7792 (VQC 36) | 0.267 | 0.156 ± 0.113 | 0.057 |
| hybrid: zz (γ=0.05; at γ=1: F1 0.018) | 6 | 7792 (VQC 36) | 0.323 | 0.201 ± 0.035 | 0.204 |
| hybrid: amplitude (γ=1; at γ=1: F1 0.857) | 8 | 326793 (VQC 48) | 0.857 | 0.857 ± 0.005 | 0.098 |
| hybrid: angle (γ=0.5; at γ=1: F1 0.243) | 8 | 10386 (VQC 48) | 0.401 | 0.282 ± 0.090 | 0.159 |
| hybrid: zz (γ=0.05; at γ=1: F1 0.019) | 8 | 10386 (VQC 48) | 0.288 | 0.161 ± 0.008 | 0.699 |

## fashion-mnist

| Model | qubits / dim | Trainable params | Accuracy | Macro-F1 | ms / image (head only) |
|---|---|---|---|---|---|
| classical: linear_bottleneck | 4 | 5174 | 0.840 | 0.841 ± 0.020 | 0.001 |
| classical: linear_bottleneck | 6 | 7756 | 0.856 | 0.856 ± 0.002 | 0.001 |
| classical: linear_bottleneck | 8 | 10338 | 0.853 | 0.852 ± 0.003 | 0.001 |
| classical: linear_bottleneck | 15 | 19375 | 0.851 | 0.851 ± 0.003 | 0.001 |
| classical: linear_bottleneck | 63 | 81343 | 0.825 | 0.824 ± 0.012 | 0.002 |
| classical: linear_bottleneck | 255 | 329215 | 0.807 | 0.808 ± 0.016 | 0.008 |
| classical: logreg_1280 | 1280 | 12810 | 0.856 | 0.856 | 0.004 |
| classical: mlp | 4 | 5194 | 0.511 | 0.439 ± 0.061 | 0.001 |
| classical: mlp | 6 | 7798 | 0.767 | 0.736 ± 0.010 | 0.001 |
| classical: mlp | 8 | 10410 | 0.772 | 0.745 ± 0.012 | 0.001 |
| hybrid: amplitude (γ=2; at γ=1: F1 0.859) | 4 | 19289 (VQC 24) | 0.865 | 0.863 ± 0.003 | 0.025 |
| hybrid: angle (γ=0.5; at γ=1: F1 0.069) | 4 | 5198 (VQC 24) | 0.281 | 0.188 ± 0.240 | 0.029 |
| hybrid: zz (γ=0.5; at γ=1: F1 0.026) | 4 | 5198 (VQC 24) | 0.294 | 0.166 ± 0.129 | 0.065 |
| hybrid: amplitude (γ=2; at γ=1: F1 0.869) | 6 | 80809 (VQC 36) | 0.873 | 0.873 ± 0.002 | 0.038 |
| hybrid: angle (γ=0.25; at γ=1: F1 0.255) | 6 | 7792 (VQC 36) | 0.380 | 0.267 ± 0.007 | 0.054 |
| hybrid: zz (γ=0.5; at γ=1: F1 0.019) | 6 | 7792 (VQC 36) | 0.298 | 0.177 ± 0.150 | 0.196 |
| hybrid: amplitude (γ=0.25; at γ=1: F1 0.871) | 8 | 326793 (VQC 48) | 0.873 | 0.873 ± 0.002 | 0.102 |
| hybrid: angle (γ=1; at γ=1: F1 0.201) | 8 | 10386 (VQC 48) | 0.327 | 0.201 ± 0.048 | 0.160 |
| hybrid: zz (γ=0.05; at γ=1: F1 0.018) | 8 | 10386 (VQC 48) | 0.344 | 0.197 ± 0.074 | 0.683 |

## Method A: correlation of a-priori metrics with trained test F1

Spearman ρ over all (encoding, γ) configurations within each dataset × qubit count, then averaged.

| Metric | mean ρ | min ρ | max ρ |
|---|---|---|---|
| kta | 0.78 | 0.73 | 0.82 |
| hs | 0.21 | 0.13 | 0.32 |
| probe | 0.25 | 0.00 | 0.62 |
| combo | 0.48 | 0.41 | 0.61 |
| var | 0.11 | -0.02 | 0.24 |

## Method A: choosing encoding and γ without training

Oracle = best test F1 over all trained configs. Regret = oracle F1 − F1 of the chosen config. Short training = pick config by validation F1 after 1 or 2 epochs (needs every config trained that long).

| Selector | top-1 encoding hit rate | mean regret (F1) | max regret | cost (s, all configs) |
|---|---|---|---|---|
| kta | 100% | 0.005 | 0.009 | 11 |
| hs | 0% | 0.691 | 0.798 | 11 |
| probe | 33% | 0.449 | 0.718 | 11 |
| combo | 17% | 0.559 | 0.733 | 11 |
| short training (1 epoch) | 100% | 0.004 | 0.008 | 66 |
| short training (2 epochs) | 100% | 0.003 | 0.007 | 133 |
| always angle, γ=1 | 0% | 0.704 | 0.812 | 0 |
| full training (val F1) | 100% | 0.002 | 0.004 | 663 |

Per case (dataset, qubits): oracle vs choice of `kta` and of 1-epoch training

| dataset | n | oracle | kta picks | 1-epoch picks |
|---|---|---|---|---|
| cifar10 | 4 | amplitude γ=0.5 (F1 0.844) | amplitude γ=2 (−0.009) | amplitude γ=0.5 (−0.000) |
| cifar10 | 6 | amplitude γ=0.1 (F1 0.856) | amplitude γ=2 (−0.005) | amplitude γ=0.5 (−0.006) |
| cifar10 | 8 | amplitude γ=0.05 (F1 0.859) | amplitude γ=2 (−0.006) | amplitude γ=1 (−0.001) |
| fashion-mnist | 4 | amplitude γ=0.25 (F1 0.867) | amplitude γ=2 (−0.004) | amplitude γ=1 (−0.008) |
| fashion-mnist | 6 | amplitude γ=2 (F1 0.873) | amplitude γ=2 (−0.000) | amplitude γ=2 (−0.000) |
| fashion-mnist | 8 | amplitude γ=0.1 (F1 0.878) | amplitude γ=2 (−0.008) | amplitude γ=1 (−0.006) |

## Method A: choosing γ for a fixed encoding

| Selector | mean regret (F1) | max regret |
|---|---|---|
| kta | 0.017 | 0.119 |
| hs | 0.032 | 0.133 |
| probe | 0.020 | 0.127 |
| combo | 0.018 | 0.119 |
| fixed γ=1 | 0.064 | 0.183 |