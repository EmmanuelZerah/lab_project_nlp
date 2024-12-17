#!/bin/bash
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --time=18:00:00
#SBATCH --job-name=two_coins_medium_with_p05
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/run_logs/two_coins_medium_with_p05.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/run_logs/two_coins_medium_with_p05.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

python coins_exp.py -t two_coins_medium_p05-00 -p 0.5 0.0 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-001 -p 0.5 0.01 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-005 -p 0.5 0.05 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-01 -p 0.5 0.1 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-02 -p 0.5 0.2 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-03 -p 0.5 0.3 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-04 -p 0.5 0.4 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-06 -p 0.5 0.6 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-07 -p 0.5 0.7 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-08 -p 0.5 0.8 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-09 -p 0.5 0.9 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-095 -p 0.5 0.95 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-099 -p 0.5 0.99 -m gpt-medium
python coins_exp.py -t two_coins_medium_p05-1 -p 0.5 1.0 -m gpt-medium