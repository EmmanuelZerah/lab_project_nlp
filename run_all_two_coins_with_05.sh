#!/bin/bash
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --time=12:00:00
#SBATCH --job-name=two_coins_more_samples_with_p10
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/logs/two_coins_more_samples_with_p10.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/logs/two_coins_more_samples_with_p10.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

python coins_exp.py -t two_coins_more_samples_p10-00 -p 1.0 0.0 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-005 -p 1.0 0.05 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-01 -p 1.0 0.1 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-015 -p 1.0 0.15 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-02 -p 1.0 0.2 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-025 -p 1.0 0.25 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-03 -p 1.0 0.3 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-035 -p 1.0 0.35 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-04 -p 1.0 0.4 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-045 -p 1.0 0.45 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-05 -p 1.0 0.5 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-055 -p 1.0 0.55 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-06 -p 1.0 0.6 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-065 -p 1.0 0.65 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-07 -p 1.0 0.7 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-075 -p 1.0 0.75 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-08 -p 1.0 0.8 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-085 -p 1.0 0.85 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-09 -p 1.0 0.9 -m gpt2-s 39
python coins_exp.py -t two_coins_more_samples_p10-095 -p 1.0 0.95 -m gpt2-s 39