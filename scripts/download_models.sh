#!/bin/bash
#SBATCH --job-name=download_models
#SBATCH --output=logs/array_er%A_%a.out
#SBATCH --error=logs/array_er%A_%a.err
#SBATCH --time=240:00

#SBATCH --partition=general
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-user=chriswolfram@uchicago.edu
#SBATCH --mail-type=END,FAIL

UNIREPS_DIR="/home/chriswolfram/unireps"
UNIREPS_HF_CACHE="/net/scratch2/chriswolfram/unireps/hf_cache"
UNIREPS_DATASETS="/net/scratch2/chriswolfram/unireps/datasets"
UNIREPS_OUTPUTS="/net/scratch2/chriswolfram/unireps/outputs"

UNIREPS_PYTHON="$UNIREPS_DIR/.venv/bin/python"

UNIREPS_PYTHON $UNIREPS_DIR/download_models.py $UNIREPS_DIR/models.txt $UNIREPS_HF_CACHE
