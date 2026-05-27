#!/bin/bash
source /data/twinkle/anaconda3/etc/profile.d/conda.sh
conda activate DR_classification
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com
cd /home/twinkle/app/LLM_paper/.worktrees/dr-classification/DR_classification
echo "env: $(which python)"
echo "torch cuda: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "starting training..."
python -u train_severity.py
