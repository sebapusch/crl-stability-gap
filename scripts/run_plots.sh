#!/bin/bash

ALGORITHMS=("sacd")
OPTIMIZERS=("adam" "rmsprop" "adamw")
LEARNING_RATES=("00001" "000025" "0001" "00025")
SEEDS="0 1 2 3 4 5 6 7 8 9"

REMOTE_HOST="s5488079@login1.hb.hpc.rug.nl"
REMOTE_BASE="/scratch/s5488079/crl-stability-gap/output"

for algo in "${ALGORITHMS[@]}"; do
    for opt in "${OPTIMIZERS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            echo "=== Processing $algo $opt $lr ==="

            remote_pattern="${REMOTE_BASE}/${algo}_cp_ji_optim/${algo}_cp_ji_optim-s*-o_${opt}-l_${lr}-*.csv"
            local_dir="./output/output/${algo}_cp_ji_${opt}_${lr}"
            mkdir -p "$local_dir"
            echo "Copying files matching: $remote_pattern"
            scp -r "${REMOTE_HOST}:${remote_pattern}" "$local_dir/"

            method_path="${algo}_cp_ji_${opt}_${lr}/${algo}_cp_ji_optim-s_<s>-o_${opt}-l_${lr}"
            prefix="${algo}_${opt}_${lr}"
            output_dir="${algo}_${opt}_${lr}"

echo "Running plot_iqm.py..."
            python scripts/plot_iqm.py \
                --env_name CartPole \
                --seeds $SEEDS \
                --methods "$method_path" \
                --prefix "$prefix" \
                --output_dir "$output_dir"

            echo "--- Done with $algo $opt $lr ---"
            echo ""
        done
    done
done

echo "All jobs complete!"