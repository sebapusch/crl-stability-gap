sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 1 --project cartpole_full_replay --method sequential --name_prefix sequential-s1
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 2 --project cartpole_full_replay --method sequential --name_prefix sequential-s2
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 3 --project cartpole_full_replay --method sequential --name_prefix sequential-s3
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 4 --project cartpole_full_replay --method sequential --name_prefix sequential-s4
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 5 --project cartpole_full_replay --method sequential --name_prefix sequential-s5

sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 1 --project cartpole_full_replay --method fine_tune --name_prefix fine_tune-s1
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 2 --project cartpole_full_replay --method fine_tune --name_prefix fine_tune-s2
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 5 --project cartpole_full_replay --method fine_tune --name_prefix fine_tune-s5
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 3 --project cartpole_full_replay --method fine_tune --name_prefix fine_tune-s3
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 4 --project cartpole_full_replay --method fine_tune --name_prefix fine_tune-s4

sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 1 --project cartpole_full_replay --method continual --name_prefix continual-s1
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 2 --project cartpole_full_replay --method continual --name_prefix continual-s2
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 3 --project cartpole_full_replay --method continual --name_prefix continual-s3
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 4 --project cartpole_full_replay --method continual --name_prefix continual-s4
sbatch dispatch/dispatch_cartpole.sh --time=00:30:00 --seed 5 --project cartpole_full_replay --method continual --name_prefix continual-s5