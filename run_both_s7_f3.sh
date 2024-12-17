#!/bin/bash
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --time=18:00:00
#SBATCH --job-name=dice_both_s7_f3
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/run_logs/dice_both_s7_f3.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/run_logs/dice_both_s7_f3.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

python dice_exp.py -t dice_both_s7_f3 -s 7 -f 3 -d both -m gpt2