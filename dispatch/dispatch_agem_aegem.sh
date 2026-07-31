#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Easy benchmark (V1, V2, V3)
python dispatch/dispatch_yaml.py dispatch/experiments/cartpole/dqn_cp_agem.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments/cartpole/dqn_cp_aegem.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments/cartpole/sacd_cp_agem_ent.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments/cartpole/sacd_cp_aegem_ent.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments/inverted_pendulum/ddpg_ip_agem.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments/inverted_pendulum/ddpg_ip_aegem.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments/inverted_pendulum/sac_ip_agem_ent.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments/inverted_pendulum/sac_ip_aegem_ent.yaml "$@"

# Hard benchmark (V2, V8, V9)
python dispatch/dispatch_yaml.py dispatch/experiments_2-8-9/cartpole/dqn_cp_agem_v2.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments_2-8-9/cartpole/dqn_cp_aegem_v2.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments_2-8-9/cartpole/sacd_cp_agem_ent_v2.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments_2-8-9/cartpole/sacd_cp_aegem_ent_v2.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments_2-8-9/inverted_pendulum/ddpg_ip_agem_v2.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments_2-8-9/inverted_pendulum/ddpg_ip_aegem_v2.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments_2-8-9/inverted_pendulum/sac_ip_agem_ent_v2.yaml "$@"
python dispatch/dispatch_yaml.py dispatch/experiments_2-8-9/inverted_pendulum/sac_ip_aegem_ent_v2.yaml "$@"
