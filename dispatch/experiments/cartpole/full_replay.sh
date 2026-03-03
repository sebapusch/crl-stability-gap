sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 1 --project cartpole_full_replay --method sequential --name_prefix sequential-s1
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 2 --project cartpole_full_replay --method sequential --name_prefix sequential-s2
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 3 --project cartpole_full_replay --method sequential --name_prefix sequential-s3
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 4 --project cartpole_full_replay --method sequential --name_prefix sequential-s4
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 5 --project cartpole_full_replay --method sequential --name_prefix sequential-s5

sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 1 --project cartpole_full_replay --method fine_tune --name_prefix fine_tune-s1
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 2 --project cartpole_full_replay --method fine_tune --name_prefix fine_tune-s2
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 5 --project cartpole_full_replay --method fine_tune --name_prefix fine_tune-s5
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 3 --project cartpole_full_replay --method fine_tune --name_prefix fine_tune-s3
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 4 --project cartpole_full_replay --method fine_tune --name_prefix fine_tune-s4

sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 1 --project cartpole_full_replay --method continual --name_prefix continual-s1
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 2 --project cartpole_full_replay --method continual --name_prefix continual-s2
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 3 --project cartpole_full_replay --method continual --name_prefix continual-s3
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 4 --project cartpole_full_replay --method continual --name_prefix continual-s4
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 5 --project cartpole_full_replay --method continual --name_prefix continual-s5

sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 1 --project cartpole_full_replay --method continual --name_prefix continual_encode-s1 --encode_task
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 2 --project cartpole_full_replay --method continual --name_prefix continual_encode-s2 --encode_task
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 3 --project cartpole_full_replay --method continual --name_prefix continual_encode-s3 --encode_task
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 4 --project cartpole_full_replay --method continual --name_prefix continual_encode-s4 --encode_task
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 5 --project cartpole_full_replay --method continual --name_prefix continual_encode-s5 --encode_task


sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 1 --project cartpole_full_replay --method continual --name_prefix continual_balanced-s1 --balanced_sampling
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 2 --project cartpole_full_replay --method continual --name_prefix continual_balanced-s2 --balanced_sampling
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 3 --project cartpole_full_replay --method continual --name_prefix continual_balanced-s3 --balanced_sampling
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 4 --project cartpole_full_replay --method continual --name_prefix continual_balanced-s4 --balanced_sampling
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 5 --project cartpole_full_replay --method continual --name_prefix continual_balanced-s5 --balanced_sampling

sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 1 --project cartpole_full_replay --method continual --name_prefix continual_balanced_encode-s1 --balanced_sampling --encode_task
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 2 --project cartpole_full_replay --method continual --name_prefix continual_balanced_encode-s2 --balanced_sampling --encode_task
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 3 --project cartpole_full_replay --method continual --name_prefix continual_balanced_encode-s3 --balanced_sampling --encode_task
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 4 --project cartpole_full_replay --method continual --name_prefix continual_balanced_encode-s4 --balanced_sampling --encode_task
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 5 --project cartpole_full_replay --method continual --name_prefix continual_balanced_encode-s5 --balanced_sampling --encode_task

sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 1 --project cartpole_full_replay --method continual --name_prefix continual_balanced_encode_inverse-s1 --balanced_sampling --encode_task --benchmark V3 V2 V1
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 2 --project cartpole_full_replay --method continual --name_prefix continual_balanced_encode_inverse-s2 --balanced_sampling --encode_task --benchmark V3 V2 V1
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 3 --project cartpole_full_replay --method continual --name_prefix continual_balanced_encode_inverse-s3 --balanced_sampling --encode_task --benchmark V3 V2 V1
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 4 --project cartpole_full_replay --method continual --name_prefix continual_balanced_encode_inverse-s4 --balanced_sampling --encode_task --benchmark V3 V2 V1
sbatch --time=01:30:00 dispatch/dispatch_cartpole.sh --seed 5 --project cartpole_full_replay --method continual --name_prefix continual_balanced_encode_inverse-s5 --balanced_sampling --encode_task --benchmark V3 V2 V1