#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=1:0:0
#$ -S /bin/bash
#$ -N predict_pseudo_ct
#$ -l gpu=true

hostname
date
nvidia-smi

source ~/challenge/venv/bin/activate

cd ~/challenge/baseline

python predict.py \
    --features_dir ~/challenge/bic-mac-data/train/sub-000/features/ \
    --output_ct ./pseudo-ct-try.nii.gz

date
