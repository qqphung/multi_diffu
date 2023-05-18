#!/bin/bash
#SBATCH --job-name=draw_ml_loss1500_300
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
python sample_chatgpt_mod_loss.py --foldername counting --type counting --data drawbench --workspace drawbench_loss
python sample_chatgpt_mod_loss.py --foldername spatial --type spatial --data drawbench --workspace drawbench_loss
