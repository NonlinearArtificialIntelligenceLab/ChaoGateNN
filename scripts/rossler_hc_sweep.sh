#!/bin/bash

# List of logic gates
gates=("AND" "OR" "XOR" "NAND" "NOR" "XNOR")

# Number of seeds
NUM_SEEDS=3
PARALLEL_JOBS=5
PLATFORM="cpu"

ROSSLER_HC_SCRIPT="./scripts/rossler_hc_chaogate.py"
OUTPUT_BASE_DIR="./output"



# Iterate over each gate
for gate in "${gates[@]}"; do
    # Iterate over each seed
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
        for c in $(seq 0 0.01 0.007); do
            out_path="${OUTPUT_BASE_DIR}/rossler_hc/${gate}/c${c}_seed${seed}/"
            echo "Running rossler_hc_chaogate.py for gate $gate with seed $seed and c=$c"
            JAX_PLATFORM_NAME=$PLATFORM python $ROSSLER_HC_SCRIPT --gate $gate --seed $seed --c $c --out_path $out_path &
        done
        if (( $((seed % PARALLEL_JOBS)) == 0 )); then
            echo "Waiting for current jobs to finish..."
            wait
        fi
    done
done

wait

echo "All jobs completed."