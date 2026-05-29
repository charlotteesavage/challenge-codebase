#!/bin/bash
#$ -N cs_model_train_check
#$ -o logs/train_check.log
#$ -e logs/train_check.err
#$ -l gpu=true
#$ -l h_vmem=32G
#$ -l h_rt=6:0:0

hostname
date
nvidia-smi

source /share/apps/source_files/python/python-3.12.11.source
source ~/challenge/venv/bin/activate

set -e

mkdir -p logs

echo "=== cs_model train_check ==="
echo "Start time: $(date)"
echo "Host: $(hostname)"

cd ~/challenge/cs_model/

python train_check.py

echo "End time: $(date)"
