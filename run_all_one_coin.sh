#!/bin/bash
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:43g
#SBATCH --time=12:00:00
#SBATCH --job-name=all_one_coin_more_samples_7
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/logs/all_one_coin_more_samples_7.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/logs/all_one_coin_more_samples_7.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

python3 coins_exp.py -t one_coin_more_samples_7_p00 -p 0.0 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p005 -p 0.05 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p01 -p 0.1 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p015 -p 0.15 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p02 -p 0.2 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p025 -p 0.25 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p03 -p 0.3 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p035 -p 0.35 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p04 -p 0.4 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p045 -p 0.45 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p05 -p 0.5 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p055 -p 0.55 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p06 -p 0.6 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p065 -p 0.65 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p07 -p 0.7 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p075 -p 0.75 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p08 -p 0.8 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p085 -p 0.85 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p09 -p 0.9 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p095 -p 0.95 -m gpt2 -s 43
python3 coins_exp.py -t one_coin_more_samples_7_p10 -p 1.0 -m gpt2 -s 43

