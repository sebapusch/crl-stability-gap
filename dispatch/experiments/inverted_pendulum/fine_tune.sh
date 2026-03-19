sbatch --time=01:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 0 \
  --method fine_tune \
  --name_prefix fine_tune-s0 \
  --project sac_ft --encode_task \
  --buffer_size 1_000_000 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 2500

sbatch --time=01:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 1 \
  --method fine_tune \
  --name_prefix fine_tune-s1 \
  --project sac_ft --encode_task \
  --buffer_size 1_000_000 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 2500

sbatch --time=01:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 2 \
  --method fine_tune \
  --name_prefix fine_tune-s2 \
  --project sac_ft --encode_task \
  --buffer_size 1_000_000 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 2500

sbatch --time=01:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 3 \
  --method fine_tune \
  --name_prefix fine_tune-s3 \
  --project sac_ft --encode_task \
  --buffer_size 1_000_000 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 2500

sbatch --time=01:30:00 dispatch/dispatch_projection.sh \
  --env inverted_pendulum \
  --seed 4 \
  --method fine_tune \
  --name_prefix fine_tune-s4 \
  --project sac_ft --encode_task \
  --buffer_size 1_000_000 \
  --batch_size 256 \
  --total_timesteps 30_000 \
  --eval_freq 2500

