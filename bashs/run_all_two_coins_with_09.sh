#!/bin/bash
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --time=12:00:00
#SBATCH --job-name=two_coins_more_samples_7_with_p09
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/logs/two_coins_more_samples_7_with_p09.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/logs/two_coins_more_samples_7_with_p09.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

python coins_exp.py -t two_coins_more_samples_7_p09-00 -p 0.9 0.0 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-005 -p 0.9 0.05 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-01 -p 0.9 0.1 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-015 -p 0.9 0.15 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-02 -p 0.9 0.2 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-025 -p 0.9 0.25 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-03 -p 0.9 0.3 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-035 -p 0.9 0.35 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-04 -p 0.9 0.4 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-045 -p 0.9 0.45 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-05 -p 0.9 0.5 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-055 -p 0.9 0.55 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-06 -p 0.9 0.6 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-065 -p 0.9 0.65 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-07 -p 0.9 0.7 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-075 -p 0.9 0.75 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-08 -p 0.9 0.8 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-085 -p 0.9 0.85 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-095 -p 0.9 0.95 -m gpt2 -s 43
python coins_exp.py -t two_coins_more_samples_7_p09-1 -p 0.9 1.0 -m gpt2 -s 43


