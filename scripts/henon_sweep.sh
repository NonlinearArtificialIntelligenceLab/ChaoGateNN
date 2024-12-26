#!/bin/bash

# List of logic gates
gates=("AND" "OR" "XOR" "NAND" "NOR" "XNOR")

# Number of seeds
NUM_SEEDS=10
PARALLEL_JOBS=1
PLATFORM="cpu"

HENON_SCRIPT="./scripts/henon_chaogate.py"
OUTPUT_BASE_DIR="./output"


# Iterate over each gate
for gate in "${gates[@]}"; do
    # Iterate over each seed
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
        for a in $(seq 1 0.05 1.5); do
            out_path="${OUTPUT_BASE_DIR}/henon/${gate}/a${a}_seed${seed}/"
            echo "Running henon_chaogate.py for gate $gate with seed $seed and a=$a"
            JAX_PLATFORM_NAME=$PLATFORM python $HENON_SCRIPT --gate $gate --seed $seed --a $a --out_path $out_path &
        done
        if (( $((seed % PARALLEL_JOBS)) == 0 )); then
            echo "Waiting for current jobs to finish..."
            wait
        fi
    done
done

wait

echo "All jobs completed."