#!/bin/bash

# List of logic gates
gates=("AND" "OR" "XOR" "NAND" "NOR" "XNOR")

# Number of seeds
NUM_SEEDS=10
PARALLEL_JOBS=1
PLATFORM="cpu"

DUFFING_SCRIPT="./scripts/duffing_newchaogate.py"
OUTPUT_BASE_DIR="./output"

# Number of steps for parameter 'a'
num_steps=50

# Iterate over each gate
for gate in "${gates[@]}"; do
    # Iterate over each seed
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
        for beta in $(seq 0 0.5 5); do
            out_path="${OUTPUT_BASE_DIR}/duffing_newgate/${gate}/beta${beta}_seed${seed}/"
            echo "Running duffing_newchaogate.py for gate $gate with seed $seed and beta=$beta"
            JAX_PLATFORM_NAME=$PLATFORM python $DUFFING_SCRIPT --gate $gate --seed $seed --beta $beta --out_path $out_path &
        done
        if (( $((seed % PARALLEL_JOBS)) == 0 )); then
            echo "Waiting for current jobs to finish..."
            wait
        fi
    done
done

wait

echo "All jobs completed."