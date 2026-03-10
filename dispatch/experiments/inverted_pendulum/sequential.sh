sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --seed 0 \
  --method sequential \
  --name_prefix sequential-s0 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --batch_size 256 \
  --total_timesteps 80_000

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --seed 1 \
  --method sequential \
  --name_prefix sequential-s1 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --batch_size 256 \
  --total_timesteps 80_000

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --seed 2 \
  --method sequential \
  --name_prefix sequential-s2 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --batch_size 256 \
  --total_timesteps 80_000

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --seed 3 \
  --method sequential \
  --name_prefix sequential-s3 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --batch_size 256 \
  --total_timesteps 80_000

sbatch --time=01:00:00 dispatch/dispatch_projection.sh \
  --seed 4 \
  --method sequential \
  --name_prefix sequential-s4 \
  --project sac_ip --encode_task \
  --buffer_size 1_000_000 \
  --batch_size 256 \
  --total_timesteps 80_000

