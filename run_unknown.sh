#!/bin/bash
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --time=12:00:00
#SBATCH --job-name=dice_unknown_3
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/logs/dice_unknown_3.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/logs/dice_unknown_3.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

python dice_exp.py -t dice_unknown_3_s2 -s 2 -d first_unknown -m gpt2 -r 39
python dice_exp.py -t dice_unknown_3_s3 -s 3 -d first_unknown -m gpt2 -r 39
python dice_exp.py -t dice_unknown_3_s4 -s 4 -d first_unknown -m gpt2 -r 39
python dice_exp.py -t dice_unknown_3_s5 -s 5 -d first_unknown -m gpt2 -r 39
python dice_exp.py -t dice_unknown_3_s6 -s 6 -d first_unknown -m gpt2 -r 39
python dice_exp.py -t dice_unknown_3_s7 -s 7 -d first_unknown -m gpt2 -r 39
python dice_exp.py -t dice_unknown_3_s8 -s 8 -d first_unknown -m gpt2 -r 39
python dice_exp.py -t dice_unknown_3_s9 -s 9 -d first_unknown -m gpt2 -r 39
python dice_exp.py -t dice_unknown_3_s10 -s 10 -d first_unknown -m gpt2 -r 39
python dice_exp.py -t dice_unknown_3_s11 -s 11 -d first_unknown -m gpt2 -r 39
python dice_exp.py -t dice_unknown_3_s12 -s 12 -d first_unknown -m gpt2 -r 39
