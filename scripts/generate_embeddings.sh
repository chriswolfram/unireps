#!/bin/bash
#SBATCH --job-name=generate_embeddings
#SBATCH --output=logs/array_er%A_%a.out
#SBATCH --error=logs/array_er%A_%a.err
#SBATCH --time=12:00:00

#SBATCH --partition=general
#SBATCH --mem=80G
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100|h100"
#SBATCH --mail-user=chriswolfram@uchicago.edu
#SBATCH --mail-type=END,FAIL

CODE_DIR="/home/chriswolfram/unireps"
UNIREPS_DIR="/net/scratch2/chriswolfram/unireps"

UNIREPS_PYTHON="$CODE_DIR/.venv/bin/python"

UNIREPS_HF_CACHE="$UNIREPS_DIR/hf_cache"
UNIREPS_DATASETS="$UNIREPS_DIR/datasets"
UNIREPS_OUTPUTS="$UNIREPS_DIR/outputs"
UNIREPS_TMP="$UNIREPS_DIR/tmp"

mkdir $UNIREPS_TMP
TMPDIR=$UNIREPS_TMP

$UNIREPS_PYTHON $CODE_DIR/generate_embeddings.py $UNIREPS_HF_CACHE $UNIREPS_DATASETS $UNIREPS_OUTPUTS
