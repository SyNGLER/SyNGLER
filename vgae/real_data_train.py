import os
from pathlib import Path
import argparse
import subprocess
import pandas as pd
import time 
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
os.chdir(BASE_DIR)
print(f"[cwd set] {BASE_DIR}")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="youtube",
                    help="dataset name (used to pick data dir). Default: youtube. Choices: youtube, yelp, dblp, polblogs.")
args = parser.parse_args()

dataset = args.dataset
r_list = [2,3,4,5,6,16] 
seeds = range(1)

sparse = 0.0
tau = 0.0
model = 'VGAE'
num_epoch = 500
learning_rate = 0.01

base_data_dir = f'../datasets/{dataset}/generator/'
output_base_dir = f'../synthetic/{dataset}/vgae-sample'

lcc_path = os.path.join(base_data_dir, 'lcc.csv')
lcc_df = pd.read_csv(lcc_path, header=None, names=['seed', 'n'])
for r in r_list:
    for seed in seeds:
        data_path = os.path.join(base_data_dir, f'seed={seed}.npy')
        output_dir = os.path.join(output_base_dir, f'r={r}', f'seed={seed}')
        
        os.makedirs(output_dir, exist_ok=True)
        try:
            n_lcc = int(lcc_df.loc[lcc_df['seed'] == seed, 'n'].values[0])
        except IndexError:
            print(f"Not Found: n for seed = {seed}. Skipping this combination.")
            continue
        command = [
            'python', 'train_real_data.py',
            '--model', model,
            '--data_path', data_path,
            '--output_dir', output_dir,
            '--input_dim', str(n_lcc),
            '--r', str(r),
            '--num_epoch', str(num_epoch),
            '--learning_rate', str(learning_rate),
            '--hidden2_dim', str(r)
        ]
        start_time = time.time()
        try:
            subprocess.run(command, check=True)
            elapsed = time.time() - start_time
            print(f"Completed: r={r}, seed={seed} in {elapsed:.2f} seconds.")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"Error: r={r}, seed={seed} failed after {elapsed:.2f} seconds.")