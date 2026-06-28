#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err

cd /scratch/$USER/crl-stability-gap
source .venv/bin/activate
module load CUDA/12.6.0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python scripts/linear_interpolation_grid.py "$@"
