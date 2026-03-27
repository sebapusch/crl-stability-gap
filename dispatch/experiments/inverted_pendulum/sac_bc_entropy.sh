# entropy 0

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e0-s0 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e0-s1 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e0-s2 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e0-s3 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e0-s4 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0

# entropy 0.01

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e001-s0 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e001-s1 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e001-s2 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e001-s3 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e001-s4 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.01

# entropy 0.1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e01-s0 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e01-s1 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e01-s2 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e01-s3 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e01-s4 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.1

# entropy 0.5

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e05-s0 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e05-s1 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e05-s2 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e05-s3 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e05-s4 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 0.5

# entropy 1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 0 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e1-s0 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 1 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e1-s1 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 2 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e1-s2 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 3 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e1-s3 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1

sbatch --time=03:00:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 4 \
  --method behavior_cloning \
  --name_prefix sac_bc_entropy_entropy-e1-s4 \
  --project sac_bc_entropy --encode_task \
  --buffer_size 1_000_000 \
  --expert_buffer_size 10_000 \
  --behavior_cloning_coefficient 0.1 \
  --batch_size 256 \
  --total_timesteps 50_000 \
  --eval_freq 5000 50 500 \
  --ent_coef 1
