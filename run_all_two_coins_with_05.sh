#!/bin/bash
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --time=16:00:00
#SBATCH --job-name=all_two_coins_with_05
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/run_logs/all_two_coins_with_05.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/run_logs/all_two_coins_with_05.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project_venv/bin/activate
python coins_exp.py -t two_coins_p05-00 -p 0.5 0.0
python coins_exp.py -t two_coins_p05-001 -p 0.5 0.01
python coins_exp.py -t two_coins_p05-005 -p 0.5 0.05
python coins_exp.py -t two_coins_p05-01 -p 0.5 0.1
python coins_exp.py -t two_coins_p05-02 -p 0.5 0.2
python coins_exp.py -t two_coins_p05-03 -p 0.5 0.3
python coins_exp.py -t two_coins_p05-04 -p 0.5 0.4
python coins_exp.py -t two_coins_p05-06 -p 0.5 0.6
python coins_exp.py -t two_coins_p05-07 -p 0.5 0.7
python coins_exp.py -t two_coins_p05-08 -p 0.5 0.8
python coins_exp.py -t two_coins_p05-09 -p 0.5 0.9
python coins_exp.py -t two_coins_p05-095 -p 0.5 0.95
python coins_exp.py -t two_coins_p05-099 -p 0.5 0.99
python coins_exp.py -t two_coins_p05-1 -p 0.5 1.0