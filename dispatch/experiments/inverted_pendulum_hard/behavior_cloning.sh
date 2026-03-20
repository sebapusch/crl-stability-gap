sbatch --time=04:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum_hard \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix bc_sac_iph-s0 \
  --project bc_sac_iph --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 100_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=04:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum_hard \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix bc_sac_iph-s1 \
  --project bc_sac_iph --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 100_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=04:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum_hard \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix bc_sac_iph-s2 \
  --project bc_sac_iph --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 100_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=04:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum_hard \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix bc_sac_iph-s3 \
  --project bc_sac_iph --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 100_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=04:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum_hard \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix bc_sac_iph-s4 \
  --project bc_sac_iph --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 100_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=04:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum_hard \
  --seed 5 \
  --method behavior_cloning \
  --name_prefix bc_sac_iph-s5 \
  --project bc_sac_iph --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 100_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=04:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum_hard \
  --seed 6 \
  --method behavior_cloning \
  --name_prefix bc_sac_iph-s6 \
  --project bc_sac_iph --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 100_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=04:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum_hard \
  --seed 7 \
  --method behavior_cloning \
  --name_prefix bc_sac_iph-s7 \
  --project bc_sac_iph --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 100_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=04:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum_hard \
  --seed 8 \
  --method behavior_cloning \
  --name_prefix bc_sac_iph-s8 \
  --project bc_sac_iph --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 100_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1

sbatch --time=04:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum_hard \
  --seed 9 \
  --method behavior_cloning \
  --name_prefix bc_sac_iph-s9 \
  --project bc_sac_iph --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 100_000 \
  --eval_freq 5000 50 500 \
  --benchmark V6 V2 V1
