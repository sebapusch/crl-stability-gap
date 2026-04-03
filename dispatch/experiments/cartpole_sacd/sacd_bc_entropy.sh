# entropy 0

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e0-s0 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e0-s1 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e0-s2 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e0-s3 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e0-s4 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 5 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e0-s5 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0 

  sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 6 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e0-s6 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0 

  sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 7 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e0-s7 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0 

  sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 8 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e0-s8 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0 

  sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 9 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e0-s9 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0 

# entropy 0.01

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e001-s0 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e001-s1 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e001-s2 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e001-s3 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e001-s4 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 5 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e001-s5 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 6 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e001-s6 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 7 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e001-s7 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 8 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e001-s8 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 9 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e001-s9 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01 

# entropy 0.1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e01-s0 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e01-s1 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e01-s2 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e01-s3 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e01-s4 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 5 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e01-s5 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 6 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e01-s6 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 7 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e01-s7 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 8 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e01-s8 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 9 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e01-s9 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1 

# entropy 0.5

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e05-s0 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e05-s1 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e05-s2 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e05-s3 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e05-s4 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 5 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e05-s5 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 6 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e05-s6 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 7 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e05-s7 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 8 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e05-s8 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 9 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e05-s9 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5 

# entropy 1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e1-s0 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e1-s1 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e1-s2 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e1-s3 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e1-s4 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 5 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e1-s5 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 6 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e1-s6 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 7 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e1-s7 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 8 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e1-s8 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1 

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --algorithm sacd \
  --seed 9 \
  --method behavior_cloning \
  --name_prefix sacd_c_bc_entropy-e1-s9 \
  --project sacd_c_bc_entropy --encode_task \
  --behavior_cloning_coefficient 0.1 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1 

