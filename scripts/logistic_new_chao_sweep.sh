#!/bin/bash

# List of logic gates
gates=("AND" "OR" "XOR" "NAND" "NOR" "XNOR")

# Number of seeds
NUM_SEEDS=10
PARALLEL_JOBS=1
PLATFORM="cpu"

LOGISTIC_SCRIPT="./scripts/logistic_newchaogate.py"
OUTPUT_BASE_DIR="./output"

# Number of steps for parameter 'a'
num_steps=50

# Iterate over each gate
for gate in "${gates[@]}"; do
    # Iterate over each seed
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
        for a in $(seq 0 0.1 4); do
            out_path="${OUTPUT_BASE_DIR}/logistic_newgate/${gate}/a${a}_seed${seed}/"
            echo "Running logistic_newchaogate.py for gate $gate with seed $seed and a=$a"
            JAX_PLATFORM_NAME=$PLATFORM python $LOGISTIC_SCRIPT --gate $gate --seed $seed --a $a --out_path $out_path &
        done
        if (( $((seed % PARALLEL_JOBS)) == 0 )); then
            echo "Waiting for current jobs to finish..."
            wait
        fi
    done
done

wait

echo "All jobs completed."