sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sacd_bc-s0 \
  --project sacd_bc --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --eval_freq 5000 50 500 \
  --total_timesteps 50_000

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sacd_bc-s1 \
  --project sacd_bc --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --eval_freq 5000 50 500 \
  --total_timesteps 50_000

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sacd_bc-s2 \
  --project sacd_bc --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --eval_freq 5000 50 500 \
  --total_timesteps 50_000

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sacd_bc-s3 \
  --project sacd_bc --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --eval_freq 5000 50 500 \
  --total_timesteps 50_000

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sacd_bc-s4 \
  --project sacd_bc --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --eval_freq 5000 50 500 \
  --total_timesteps 50_000

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 5 \
  --method behavior_cloning \
  --name_prefix sacd_bc-s5 \
  --project sacd_bc --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --eval_freq 5000 50 500 \
  --total_timesteps 50_000

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 6 \
  --method behavior_cloning \
  --name_prefix sacd_bc-s6 \
  --project sacd_bc --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --eval_freq 5000 50 500 \
  --total_timesteps 50_000

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 7 \
  --method behavior_cloning \
  --name_prefix sacd_bc-s7 \
  --project sacd_bc --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --eval_freq 5000 50 500 \
  --total_timesteps 50_000

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 8 \
  --method behavior_cloning \
  --name_prefix sacd_bc-s8 \
  --project sacd_bc --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --eval_freq 5000 50 500 \
  --total_timesteps 50_000

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 9 \
  --method behavior_cloning \
  --name_prefix sacd_bc-s9 \
  --project sacd_bc --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --eval_freq 5000 50 500 \
  --total_timesteps 50_000
