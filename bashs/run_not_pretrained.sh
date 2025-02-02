#!/bin/bash
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --job-name=5_not_pretrained_small
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/logs/5_not_pretrained_small.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/logs/5_not_pretrained_small.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

for num in {1..5}; do
    sbatch <<EOT
#!/bin/bash
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --time=4-00:00:00
#SBATCH --job-name=all_not_pretrained_small_${num}
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/logs/all_not_pretrained_small_${num}.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/logs/all_not_pretrained_small_${num}.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p00 -p 0.0 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p005 -p 0.05 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p01 -p 0.1 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p015 -p 0.15 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p02 -p 0.2 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p025 -p 0.25 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p03 -p 0.3 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p035 -p 0.35 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p04 -p 0.4 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p045 -p 0.45 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p05 -p 0.5 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p055 -p 0.55 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p06 -p 0.6 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p065 -p 0.65 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p07 -p 0.7 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p075 -p 0.75 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p08 -p 0.8 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p085 -p 0.85 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p09 -p 0.9 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p095 -p 0.95 -m gpt2
python3 coins_exp.py -e not_pretrained -t not_pretrained_small_${num}_p10 -p 1.0 -m gpt2
EOT
done