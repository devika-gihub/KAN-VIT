#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=2-24:00:00
#SBATCH --job-name=KAT-glaucoma-train
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/devikarg.cet/KAN_Resnet/MIL}"
ENV_DIR="${ENV_DIR:-${PROJECT_DIR}/myenv}"

cd "$PROJECT_DIR"
source "$ENV_DIR/bin/activate"

python3 main.py "$@"
