sbatch --time=01:30:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --seed 0 \
  --method sequential \
  --name_prefix sacd_seq-s0 \
  --project sacd_seq --encode_task \
  --eval_freq 5000

sbatch --time=01:30:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --seed 1 \
  --method sequential \
  --name_prefix sacd_seq-s1 \
  --project sacd_seq --encode_task \
  --eval_freq 5000

sbatch --time=01:30:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --seed 2 \
  --method sequential \
  --name_prefix sacd_seq-s2 \
  --project sacd_seq --encode_task \
  --eval_freq 5000

sbatch --time=01:30:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --seed 3 \
  --method sequential \
  --name_prefix sacd_seq-s3 \
  --project sacd_seq --encode_task \
  --eval_freq 5000

sbatch --time=01:30:00 dispatch/dispatch_projection.sh \
  --env cartpole \
  --seed 4 \
  --method sequential \
  --name_prefix sacd_seq-s4 \
  --project sacd_seq --encode_task \
  --eval_freq 5000

