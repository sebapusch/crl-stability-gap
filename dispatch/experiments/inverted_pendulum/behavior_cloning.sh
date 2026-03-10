sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sequential-s0 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 100 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 500

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sequential-s1 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 100 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 500

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sequential-s2 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 100 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 500

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sequential-s3 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 100 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 500

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sequential-s4 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 100 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 500

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 5 \
  --method behavior_cloning \
  --name_prefix sequential-s5 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 100 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 500

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 6 \
  --method behavior_cloning \
  --name_prefix sequential-s6 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 100 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 500

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 7 \
  --method behavior_cloning \
  --name_prefix sequential-s7 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 100 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 500

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 8 \
  --method behavior_cloning \
  --name_prefix sequential-s8 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 100 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 500

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 9 \
  --method behavior_cloning \
  --name_prefix sequential-s9 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 100 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 500