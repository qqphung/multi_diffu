#!/bin/bash
#SBATCH --job-name=loss_size-multi
#SBATCH --output=/vulcanscratch/chuonghm/layout-guidance/logs/cora2_%A.out
#SBATCH --error=/vulcanscratch/chuonghm/layout-guidance/logs/cora2_%A.err
#SBATCH --time=24:00:00
#SBATCH --account=abhinav
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=16gb
#SBATCH --cpus-per-task=4


source /cfarhomes/chuonghm/.zshrc
conda activate ldm_a6000
python sample_chatgpt_mod.py --data HRS --foldername color --type color --workspace HRS_evaluate
python sample_chatgpt_mod_loss.py --data HRS --foldername color --type color --workspace HRS_save_loss
