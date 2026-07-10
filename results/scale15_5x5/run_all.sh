#!/bin/bash
# Scale-15 sequential run: lossless -> R32 -> R38b (stable config: chunk-size 1, mem 0.72)
cd /home/gfy/CodeMAS_Project/sglang-kvflow
rm -rf results/scale15_5x5/lossless results/scale15_5x5/r32 results/scale15_5x5/r38b
mkdir -p results/scale15_5x5/lossless results/scale15_5x5/r32 results/scale15_5x5/r38b
echo "=== lossless ==="
bash results/scale15_5x5/launchers/run_lossless.sh > results/scale15_5x5/lossless.stdout 2>&1
echo "=== R32 ==="
bash results/scale15_5x5/launchers/run_r32.sh > results/scale15_5x5/r32.stdout 2>&1
echo "=== R38b ==="
bash results/scale15_5x5/launchers/run_r38b.sh > results/scale15_5x5/r38b.stdout 2>&1
echo "=== ALL DONE ==="
