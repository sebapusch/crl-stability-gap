sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac_3-s0 \
  --project sac_bc_3 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac-s1 \
  --project sac_bc_3 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac-s2 \
  --project sac_bc_3 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac-s3 \
  --project sac_bc_3 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac-s4 \
  --project sac_bc_3 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 5 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac-s5 \
  --project sac_bc_3 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 6 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac-s6 \
  --project sac_bc_3 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 7 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac-s7 \
  --project sac_bc_3 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 8 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac-s8 \
  --project sac_bc_3 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 9 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac-s9 \
  --project sac_bc_3 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1
