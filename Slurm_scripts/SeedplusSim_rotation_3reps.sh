#!/bin/bash 
#SBATCH -o slurm_SeedplusSim_rotation_3reps_20260408.out
#SBATCH -e slurm_SeedplusSim_rotation_3reps_20260408.err
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
source /hpc/dctrl/ks723/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python --version
cd /hpc/dctrl/ks723/Information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python SeedplusSim_rotation_3reps.py