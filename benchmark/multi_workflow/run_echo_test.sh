#!/bin/bash
#SBATCH --job-name=kvflow-echo
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00

echo "START"
echo "Node: $SLURM_JOB_NODELIST"
which python || echo "Python not found"
echo "END"
