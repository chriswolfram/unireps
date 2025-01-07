#!/bin/bash
#SBATCH --job-name=generate_embeddings
#SBATCH --output=logs/array_er%A_%a.out
#SBATCH --error=logs/array_er%A_%a.err
#SBATCH --time=12:00:00

#SBATCH --partition=general
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:4
#SBATCH --mail-user=chriswolfram@uchicago.edu
#SBATCH --mail-type=END,FAIL

# Monitor with watch -n 1 "squeue -u chriswolfram && echo \"\n\n\" && cat logs/array_er309471_4294967294.out && echo \"\n\n\" && cat logs/array_er309471_4294967294.err | sed 's/\r/\n/g' | tail -n 10"

UNIREPS_DIR="/home/chriswolfram/unireps"
UNIREPS_HF_CACHE="/net/scratch2/chriswolfram/unireps/hf_cache"
UNIREPS_DATASETS="/net/scratch2/chriswolfram/unireps/datasets"
UNIREPS_OUTPUTS="/net/scratch2/chriswolfram/unireps/outputs"

UNIREPS_PYTHON="$UNIREPS_DIR/.venv/bin/python"

$UNIREPS_PYTHON $UNIREPS_DIR/generate_embeddings.py $UNIREPS_DIR/models.txt $UNIREPS_HF_CACHE $UNIREPS_DIR/datasets.txt $UNIREPS_DATASETS $UNIREPS_OUTPUTS
