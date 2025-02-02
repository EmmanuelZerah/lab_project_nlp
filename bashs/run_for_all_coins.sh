#!/bin/bash
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --time=1:00:00
#SBATCH --job-name=7_one_coin_new_edges
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/logs/7_one_coin_new_edges.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/logs/7_one_coin_new_edges.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

for num in {1..7}; do
    sbatch <<EOT
#!/bin/bash
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH --time=4-00:00:00 
#SBATCH --job-name=one_coin_new_edge_${num}
#SBATCH --output=/cs/labs/oabend/manuz/lab_project/logs/one_coin_new_edge_${num}.out
#SBATCH --error=/cs/labs/oabend/manuz/lab_project/logs/one_coin_new_edge_${num}.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emmanuel.zerah@mail.huji.ac.il

source /cs/labs/oabend/manuz/lab_project/lab_venv/bin/activate

python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p00 -p 1 0.0 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p002 -p 1 0.02 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p004 -p 1 0.04 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p006 -p 1 0.06 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p008 -p 1 0.08 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p01 -p 1 0.1 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p012 -p 1 0.12 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p014 -p 1 0.14 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p016 -p 1 0.16 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p018 -p 1 0.18 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p02 -p 1 0.2 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p021 -p 1 0.21 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p022 -p 1 0.22 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p023 -p 1 0.23 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p024 -p 1 0.24 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p025 -p 1 0.25 -m gpt2-large

python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p075 -p 1 0.75 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p076 -p 1 0.76 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p077 -p 1 0.77 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p078 -p 1 0.78 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p079 -p 1 0.79 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p080 -p 1 0.80 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p082 -p 1 0.82 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p084 -p 1 0.84 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p086 -p 1 0.86 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p088 -p 1 0.88 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p09 -p 1 0.9 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p092 -p 1 0.92 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p094 -p 1 0.94 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p096 -p 1 0.96 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p098 -p 1 0.98 -m gpt2-large
python3 ../coins_exp.py -e new_edge_exp -t one_coin_edge_${num}_p10 -p 1 1.0 -m gpt2-large
EOT
done