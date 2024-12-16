#!/bin/bash
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --time=18:00:00
#SBATCH --job-name=all_one_coin
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/run_logs/all_one_coin.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/run_logs/all_one_coin.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project_venv/bin/activate
python coins_exp.py -t one_coin_p035 -p 0.35
python coins_exp.py -t one_coin_p04 -p 0.4
python coins_exp.py -t one_coin_p045 -p 0.45
python coins_exp.py -t one_coin_p05 -p 0.5
python coins_exp.py -t one_coin_p055 -p 0.55
python coins_exp.py -t one_coin_p06 -p 0.6
python coins_exp.py -t one_coin_p065 -p 0.65
python coins_exp.py -t one_coin_p07 -p 0.7
python coins_exp.py -t one_coin_p075 -p 0.75
python coins_exp.py -t one_coin_p08 -p 0.8
python coins_exp.py -t one_coin_p085 -p 0.85
python coins_exp.py -t one_coin_p09 -p 0.9
python coins_exp.py -t one_coin_p095 -p 0.95
python coins_exp.py -t one_coin_p099 -p 0.99
python coins_exp.py -t one_coin_p10 -p 1.0
