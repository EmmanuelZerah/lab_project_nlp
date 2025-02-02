#!/bin/bash
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --time=12:00:00
#SBATCH --job-name=dice_unknown_large_5
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/logs/dice_unknown_large_5.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/logs/dice_unknown_large_5.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

python dice_exp.py -t dice_unknown_large_5_s2 -s 2 -d first_unknown -m gpt2-large -r 45
python dice_exp.py -t dice_unknown_large_5_s3 -s 3 -d first_unknown -m gpt2-large -r 45
python dice_exp.py -t dice_unknown_large_5_s4 -s 4 -d first_unknown -m gpt2-large -r 45
python dice_exp.py -t dice_unknown_large_5_s5 -s 5 -d first_unknown -m gpt2-large -r 45
python dice_exp.py -t dice_unknown_large_5_s6 -s 6 -d first_unknown -m gpt2-large -r 45
python dice_exp.py -t dice_unknown_large_5_s7 -s 7 -d first_unknown -m gpt2-large -r 45
python dice_exp.py -t dice_unknown_large_5_s8 -s 8 -d first_unknown -m gpt2-large -r 45
python dice_exp.py -t dice_unknown_large_5_s9 -s 9 -d first_unknown -m gpt2-large -r 45
python dice_exp.py -t dice_unknown_large_5_s10 -s 10 -d first_unknown -m gpt2-large -r 45
python dice_exp.py -t dice_unknown_large_5_s11 -s 11 -d first_unknown -m gpt2-large -r 45
python dice_exp.py -t dice_unknown_large_5_s12 -s 12 -d first_unknown -m gpt2-large -r 45
