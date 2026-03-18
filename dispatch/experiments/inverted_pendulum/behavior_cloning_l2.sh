sbatch --time=02:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac_l2-s0 \
  --project sac_bc_l2 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 1 \
  --batch_size 256 \
  --total_timesteps 40_000 \
  --bc_loss_fn l2 \
  --eval_freq 2000 50 500

sbatch --time=02:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac_l2-s1 \
  --project sac_bc_l2 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 1 \
  --batch_size 256 \
  --total_timesteps 40_000 \
  --bc_loss_fn l2 \
  --eval_freq 2000 50 500

sbatch --time=02:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac_l2-s2 \
  --project sac_bc_l2 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 1 \
  --batch_size 256 \
  --total_timesteps 40_000 \
  --bc_loss_fn l2 \
  --eval_freq 2000 50 500

sbatch --time=02:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac_l2-s3 \
  --project sac_bc_l2 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 1 \
  --batch_size 256 \
  --total_timesteps 40_000 \
  --bc_loss_fn l2 \
  --eval_freq 2000 50 500

sbatch --time=02:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac_l2-s4 \
  --project sac_bc_l2 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 1 \
  --batch_size 256 \
  --total_timesteps 40_000 \
  --bc_loss_fn l2 \
  --eval_freq 2000 50 500

sbatch --time=02:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 5 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac_l2-s5 \
  --project sac_bc_l2 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 1 \
  --batch_size 256 \
  --total_timesteps 40_000 \
  --bc_loss_fn l2 \
  --eval_freq 2000 50 500

sbatch --time=02:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 6 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac_l2-s6 \
  --project sac_bc_l2 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 1 \
  --batch_size 256 \
  --total_timesteps 40_000 \
  --bc_loss_fn l2 \
  --eval_freq 2000 50 500

sbatch --time=02:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 7 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac_l2-s7 \
  --project sac_bc_l2 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 1 \
  --batch_size 256 \
  --total_timesteps 40_000 \
  --bc_loss_fn l2 \
  --eval_freq 2000 50 500

sbatch --time=02:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 8 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac_l2-s8 \
  --project sac_bc_l2 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 1 \
  --batch_size 256 \
  --total_timesteps 40_000 \
  --bc_loss_fn l2 \
  --eval_freq 2000 50 500

sbatch --time=02:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 9 \
  --method behavior_cloning \
  --name_prefix behavior_cloning_sac_l2-s9 \
  --project sac_bc_l2 --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 1 \
  --batch_size 256 \
  --total_timesteps 40_000 \
  --bc_loss_fn l2 \
  --eval_freq 2000 50 500
