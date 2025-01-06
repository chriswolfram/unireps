#!/bin/bash
#SBATCH --job-name=download_models
#SBATCH --output=logs/array_er%A_%a.out
#SBATCH --error=logs/array_er%A_%a.err
#SBATCH --time=240:00

#SBATCH --partition=general
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-user=chriswolfram@uchicago.edu
#SBATCH --mail-type=END,FAIL

/home/chriswolfram/unireps/.venv/bin/python /home/chriswolfram/unireps/download_models.py /home/chriswolfram/unireps/models.txt /net/scratch2/chriswolfram/hf_cache