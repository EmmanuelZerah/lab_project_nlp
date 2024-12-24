#!/bin/bash
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --time=12:00:00
#SBATCH --job-name=dice_known_1_s7
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/logs/dice_known_1_s7.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/logs/dice_known_1_s7.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

python dice_exp.py -t dice_known_1_s7_f1 -s 7 -f 1 -d first_known -m gpt2 -r 37
python dice_exp.py -t dice_known_1_s7_f2 -s 7 -f 2 -d first_known -m gpt2 -r 37
python dice_exp.py -t dice_known_1_s7_f3 -s 7 -f 3 -d first_known -m gpt2 -r 37
python dice_exp.py -t dice_known_1_s7_f4 -s 7 -f 4 -d first_known -m gpt2 -r 37
python dice_exp.py -t dice_known_1_s7_f5 -s 7 -f 5 -d first_known -m gpt2 -r 37
python dice_exp.py -t dice_known_1_s7_f6 -s 7 -f 6 -d first_known -m gpt2 -r 37