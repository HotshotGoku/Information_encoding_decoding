#!/bin/bash 
#SBATCH -o slurm_SEED_dataset_MORE_diffusion_20260403.out
#SBATCH -e slurm_SEED_dataset_MORE_diffusion_20260403.err
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python SeedDataset_MORE_Diffusion.py