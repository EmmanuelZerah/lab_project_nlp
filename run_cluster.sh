#!/bin/bash
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --time=2:00:00
#SBATCH --job-name=two_coins_exp_05-09
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/runs/run_logs/two_coins_exp_05-09.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/runs/run_logs/two_coins_exp_05-09.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project_venv/bin/activate
python coins_exp.py
