#!/bin/bash 
#SBATCH -o slurm_expcolortoseed_vae_Res_UNet_Diffusion_GPUs_20260408_run12_MORE_seeds_no_augmentation.out
#SBATCH -e slurm_expcolortoseed_vae_Res_UNet_Diffusion_GPUs_20260408_run12_MORE_seeds_no_augmentation.err
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python expcolortoseed_vae_train_GPUs.py 