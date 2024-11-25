#!/bin/bash

# List of logic gates
gates=("AND" "OR" "XOR" "NAND" "NOR" "XNOR")

# Number of seeds
NUM_SEEDS=10
PARALLEL_JOBS=1
PLATFORM="cpu"

LORENZ_SCRIPT="./scripts/lorenz_newchaogate.py"
OUTPUT_BASE_DIR="./output"



# Iterate over each gate
for gate in "${gates[@]}"; do
    # Iterate over each seed
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
        for rho in $(seq 14 1 28); do
            out_path="${OUTPUT_BASE_DIR}/lorenz_newgate/${gate}/rho${rho}_seed${seed}/"
            echo "Running lorenz_newchaogate.py for gate $gate with seed $seed and rho=$rho"
            JAX_PLATFORM_NAME=$PLATFORM python $LORENZ_SCRIPT --gate $gate --seed $seed --rho $rho --out_path $out_path &
        done
        if (( $((seed % PARALLEL_JOBS)) == 0 )); then
            echo "Waiting for current jobs to finish..."
            wait
        fi
    done
done

wait

echo "All jobs completed."