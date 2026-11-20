#!/usr/bin/env bash
# One seed per GPU; submit.sh releases a three-job array after online smoke.
#SBATCH --job-name=b4-b0-lunar-80e
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail
exec bash experiments/logit_distillation/b4_lunar_80e.sh train b4_to_b0_lunar_80e b4-to-b0-lunar-80e
