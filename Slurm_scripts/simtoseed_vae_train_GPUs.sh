#!/bin/bash 
#SBATCH -o slurm_simtoseed_Res_UNet_3RotReps_20260408_%a.out
#SBATCH -e slurm_simtoseed_Res_UNet_3RotReps_20260408_%a.err
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python simtoseed_vae_train_GPUs.py