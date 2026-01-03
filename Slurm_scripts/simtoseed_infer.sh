#!/bin/bash 
#SBATCH -o slurm_simtoseed_infer_20260102_%a.out
#SBATCH -e slurm_simtoseed_infer_20260102_%a.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Information_encoding_decoding
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python simtoseed_infer.py