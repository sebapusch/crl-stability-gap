#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err

export JAX_PLATFORMS=cpu
cd /scratch/$USER/crl-stability-gap
source .venv/bin/activate
module load CUDA/12.6.0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python scripts/linear_interpolation_grid.py "$@"
