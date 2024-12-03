#!/bin/bash
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:15G
#SBATCH --time=2:00:00
#SBATCH --job-name=basic_coin_exp_p1
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/runs/run_logs/basic_coin_exp_p1.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/runs/run_logs/basic_coin_exp_p1.err

source /cs/labs/oabend/manuz/lab_project_venv/bin/activate
python coin_exp_model.py
