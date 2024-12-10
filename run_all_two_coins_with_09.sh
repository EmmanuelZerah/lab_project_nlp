#!/bin/bash
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --time=16:00:00
#SBATCH --job-name=all_two_coins_with_09
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/runs/all_two_coins_with_09.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/runs/all_two_coins_with_09.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project_venv/bin/activate
python coins_exp.py -t two_coins_p09-00 -p 0.9 0.0 -m gpt2
python coins_exp.py -t two_coins_p09-001 -p 0.9 0.01 -m gpt2
python coins_exp.py -t two_coins_p09-005 -p 0.9 0.05 -m gpt2
python coins_exp.py -t two_coins_p09-01 -p 0.9 0.1 -m gpt2
python coins_exp.py -t two_coins_p09-02 -p 0.9 0.2 -m gpt2
python coins_exp.py -t two_coins_p09-03 -p 0.9 0.3 -m gpt2
python coins_exp.py -t two_coins_p09-04 -p 0.9 0.4 -m gpt2
python coins_exp.py -t two_coins_p09-05 -p 0.9 0.5 -m gpt2
python coins_exp.py -t two_coins_p09-06 -p 0.9 0.6 -m gpt2
python coins_exp.py -t two_coins_p09-07 -p 0.9 0.7 -m gpt2
python coins_exp.py -t two_coins_p09-08 -p 0.9 0.8 -m gpt2
python coins_exp.py -t two_coins_p09-095 -p 0.9 0.95 -m gpt2
python coins_exp.py -t two_coins_p09-099 -p 0.9 0.99 -m gpt2
python coins_exp.py -t two_coins_p09-1 -p 0.9 1.0 -m gpt2


