# AGEM and AEGEM evaluation

`P` below means the final aggregate performance `P(T)` reported by
`scripts/compute_metrics.py`. The evaluations use IQM aggregation over 10 seeds,
the last 5 evaluation points for smoothing, and 95% bootstrap confidence
intervals. Scores are normalized percentages.

## Best parameters

| Method | Selected parameters | P(T) | V1 | V2 | V3 | min-ACC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AEGEM | `lr=0.0001`, `behavior_cloning_coefficient=0.1` (tie-break) | **83.7 [72.6, 92.3]** | 99.7 [87.1, 100.0] | 51.5 [25.3, 82.4] | 100.0 [73.9, 100.0] | 5.9 [3.5, 7.0] |
| AGEM | `lr=0.0001` | **78.1 [54.2, 93.2]** | 82.7 [56.8, 99.0] | 71.0 [38.9, 94.7] | 80.8 [50.5, 97.0] | 5.6 [3.7, 12.4] |

AEGEM has the higher point estimate for `P(T)`, although its interval overlaps
AGEM's. AGEM performs better on V2, while AEGEM's stronger V1 and V3 scores
produce the higher aggregate score.

All four evaluated AEGEM behavior-cloning coefficients (`0.1`, `0.4`, `0.7`,
and `1.0`) tie exactly under these metrics. The smallest value, `0.1`, is used
in the paper plot configuration only as a deterministic tie-break; the data do
not support preferring it over the other tied values. The earlier AEGEM runs
whose filenames do not include the coefficient produce the same metrics.

## Parameter sweep

| Method | Behavior-cloning coefficient | Learning rate | P(T) | min-ACC |
| --- | ---: | ---: | ---: | ---: |
| AEGEM | 0.1, 0.4, 0.7, or 1.0 | 0.0001 | **83.7 [72.6, 92.3]** | 5.9 [3.5, 7.0] |
| AEGEM | 0.1, 0.4, 0.7, or 1.0 | 0.00025 | 77.4 [57.6, 91.8] | 7.7 [3.9, 19.0] |
| AGEM | n/a | 0.0001 | **78.1 [54.2, 93.2]** | 5.6 [3.7, 12.4] |
| AGEM | n/a | 0.00025 | 47.8 [37.4, 61.4] | 7.2 [4.4, 8.7] |

## Result availability

The workspace contains result CSVs only for the easy CartPole DQN experiments
(`dqn_cp_agem` and `dqn_cp_aegem`). The other AGEM/AEGEM dispatch configs have
no corresponding directory under `output/output`, so they cannot be evaluated
or ranked from this checkout:

- Easy benchmark: SACD CartPole, DDPG Inverted Pendulum, and SAC Inverted Pendulum.
- Hard benchmark (V2/V8/V9): DQN and SACD CartPole, plus DDPG and SAC Inverted Pendulum.

The result filenames belong to the historical learning-rate and
behavior-cloning-coefficient sweeps, while the current dispatch YAML files no
longer contain those ablations. The evaluation used the parameter order encoded
in the filenames (`b`, `s`, `l` for AEGEM and `s`, `l` for AGEM).

## Plot configurations

- `dqn_cp_aegem_best.yaml`: the tied AEGEM winner (`b=0.1`, `lr=0.0001`) on V1, V2, and V3.
- `dqn_cp_agem_best.yaml`: the AGEM winner (`lr=0.0001`) on V1, V2, and V3.
