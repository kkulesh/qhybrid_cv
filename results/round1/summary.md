# Round 1 results

## cifar10

| Model | qubits / dim | Trainable params | Accuracy | Macro-F1 | ms / image (head only) |
|---|---|---|---|---|---|
| classical: linear_bottleneck | 4 | 5174 | 0.739 | 0.736 ± 0.081 | 0.002 |
| classical: linear_bottleneck | 6 | 7756 | 0.828 | 0.828 ± 0.018 | 0.001 |
| classical: linear_bottleneck | 8 | 10338 | 0.826 | 0.827 ± 0.007 | 0.001 |
| classical: linear_bottleneck | 15 | 19375 | 0.833 | 0.832 ± 0.003 | 0.002 |
| classical: linear_bottleneck | 63 | 81343 | 0.814 | 0.813 ± 0.010 | 0.003 |
| classical: linear_bottleneck | 255 | 329215 | 0.789 | 0.788 ± 0.003 | 0.008 |
| classical: logreg_1280 | 1280 | 12810 | 0.844 | 0.843 | 0.003 |
| classical: mlp | 4 | 5194 | 0.552 | 0.504 ± 0.084 | 0.001 |
| classical: mlp | 6 | 7798 | 0.500 | 0.429 ± 0.285 | 0.001 |
| classical: mlp | 8 | 10410 | 0.703 | 0.695 ± 0.055 | 0.001 |
| hybrid: amplitude (γ=0.5; at γ=1: F1 0.806) | 4 | 19297 (VQC 24) | 0.801 | 0.802 ± 0.006 | 0.038 |
| hybrid: angle (γ=0.05; at γ=1: F1 0.018) | 4 | 5206 (VQC 24) | 0.581 | 0.516 ± 0.065 | 0.044 |
| hybrid: zz (γ=0.05; at γ=1: F1 0.031) | 4 | 5206 (VQC 24) | 0.295 | 0.191 ± 0.062 | 0.094 |
| hybrid: amplitude (γ=1; at γ=1: F1 0.842) | 6 | 80821 (VQC 36) | 0.842 | 0.842 ± 0.004 | 0.060 |
| hybrid: angle (γ=0.1; at γ=1: F1 0.044) | 6 | 7804 (VQC 36) | 0.635 | 0.602 ± 0.066 | 0.077 |
| hybrid: zz (γ=0.05; at γ=1: F1 0.018) | 6 | 7804 (VQC 36) | 0.438 | 0.360 ± 0.145 | 0.182 |
| hybrid: amplitude (γ=2; at γ=1: F1 0.834) | 8 | 326809 (VQC 48) | 0.835 | 0.835 ± 0.004 | 0.089 |
| hybrid: angle (γ=0.1; at γ=1: F1 0.192) | 8 | 10402 (VQC 48) | 0.638 | 0.607 ± 0.077 | 0.138 |
| hybrid: zz (γ=0.05; at γ=1: F1 0.018) | 8 | 10402 (VQC 48) | 0.509 | 0.441 ± 0.032 | 0.666 |

## fashion-mnist

| Model | qubits / dim | Trainable params | Accuracy | Macro-F1 | ms / image (head only) |
|---|---|---|---|---|---|
| classical: linear_bottleneck | 4 | 5174 | 0.840 | 0.841 ± 0.020 | 0.001 |
| classical: linear_bottleneck | 6 | 7756 | 0.856 | 0.856 ± 0.002 | 0.001 |
| classical: linear_bottleneck | 8 | 10338 | 0.853 | 0.852 ± 0.003 | 0.001 |
| classical: linear_bottleneck | 15 | 19375 | 0.851 | 0.851 ± 0.003 | 0.001 |
| classical: linear_bottleneck | 63 | 81343 | 0.825 | 0.824 ± 0.012 | 0.003 |
| classical: linear_bottleneck | 255 | 329215 | 0.807 | 0.808 ± 0.016 | 0.009 |
| classical: logreg_1280 | 1280 | 12810 | 0.856 | 0.856 | 0.004 |
| classical: mlp | 4 | 5194 | 0.511 | 0.439 ± 0.061 | 0.001 |
| classical: mlp | 6 | 7798 | 0.767 | 0.736 ± 0.010 | 0.002 |
| classical: mlp | 8 | 10410 | 0.772 | 0.745 ± 0.012 | 0.002 |
| hybrid: amplitude (γ=1; at γ=1: F1 0.852) | 4 | 19297 (VQC 24) | 0.851 | 0.852 ± 0.015 | 0.045 |
| hybrid: angle (γ=0.05; at γ=1: F1 0.152) | 4 | 5206 (VQC 24) | 0.550 | 0.453 ± 0.150 | 0.045 |
| hybrid: zz (γ=0.05; at γ=1: F1 0.083) | 4 | 5206 (VQC 24) | 0.283 | 0.170 ± 0.134 | 0.091 |
| hybrid: amplitude (γ=2; at γ=1: F1 0.844) | 6 | 80821 (VQC 36) | 0.856 | 0.855 ± 0.012 | 0.057 |
| hybrid: angle (γ=0.05; at γ=1: F1 0.178) | 6 | 7804 (VQC 36) | 0.547 | 0.462 ± 0.281 | 0.076 |
| hybrid: zz (γ=0.05; at γ=1: F1 0.042) | 6 | 7804 (VQC 36) | 0.422 | 0.329 ± 0.189 | 0.201 |
| hybrid: amplitude (γ=0.05; at γ=1: F1 0.849) | 8 | 326809 (VQC 48) | 0.858 | 0.859 ± 0.014 | 0.084 |
| hybrid: angle (γ=0.05; at γ=1: F1 0.361) | 8 | 10402 (VQC 48) | 0.668 | 0.621 ± 0.116 | 0.133 |
| hybrid: zz (γ=0.05; at γ=1: F1 0.020) | 8 | 10402 (VQC 48) | 0.460 | 0.382 ± 0.112 | 0.629 |

## Method A: correlation of a-priori metrics with trained test F1

Spearman ρ over all (encoding, γ) configurations within each dataset × qubit count, then averaged.

| Metric | mean ρ | min ρ | max ρ |
|---|---|---|---|
| kta | 0.78 | 0.67 | 0.86 |
| hs | -0.03 | -0.15 | 0.07 |
| probe | 0.38 | 0.09 | 0.66 |
| combo | 0.44 | 0.37 | 0.52 |
| var | -0.11 | -0.40 | 0.12 |

## Method A: choosing encoding and γ without training

Oracle = best test F1 over all trained configs. Regret = oracle F1 − F1 of the chosen config. Short training = pick config by validation F1 after 1 or 2 epochs (needs every config trained that long).

| Selector | top-1 encoding hit rate | mean regret (F1) | max regret | cost (s, all configs) |
|---|---|---|---|---|
| kta | 100% | 0.006 | 0.014 | 11 |
| hs | 0% | 0.686 | 0.798 | 11 |
| probe | 67% | 0.197 | 0.702 | 11 |
| combo | 67% | 0.194 | 0.702 | 11 |
| short training (1 epoch) | 100% | 0.008 | 0.016 | 73 |
| short training (2 epochs) | 100% | 0.005 | 0.009 | 145 |
| always angle, γ=1 | 0% | 0.686 | 0.798 | 0 |
| full training (val F1) | 100% | 0.003 | 0.011 | 725 |

Per case (dataset, qubits): oracle vs choice of `kta` and of 1-epoch training

| dataset | n | oracle | kta picks | 1-epoch picks |
|---|---|---|---|---|
| cifar10 | 4 | amplitude γ=0.1 (F1 0.813) | amplitude γ=2 (−0.006) | amplitude γ=0.25 (−0.007) |
| cifar10 | 6 | amplitude γ=1 (F1 0.842) | amplitude γ=2 (−0.014) | amplitude γ=0.1 (−0.013) |
| cifar10 | 8 | amplitude γ=0.1 (F1 0.841) | amplitude γ=2 (−0.005) | amplitude γ=0.1 (−0.000) |
| fashion-mnist | 4 | amplitude γ=2 (F1 0.855) | amplitude γ=2 (−0.000) | amplitude γ=0.25 (−0.005) |
| fashion-mnist | 6 | amplitude γ=2 (F1 0.855) | amplitude γ=2 (−0.000) | amplitude γ=0.05 (−0.007) |
| fashion-mnist | 8 | amplitude γ=0.05 (F1 0.859) | amplitude γ=2 (−0.008) | amplitude γ=0.5 (−0.016) |

## Method A: choosing γ for a fixed encoding

| Selector | mean regret (F1) | max regret |
|---|---|---|
| kta | 0.151 | 0.558 |
| hs | 0.158 | 0.558 |
| probe | 0.137 | 0.558 |
| combo | 0.138 | 0.558 |
| fixed γ=1 | 0.223 | 0.558 |