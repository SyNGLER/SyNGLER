import os
from pathlib import Path
import subprocess
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
os.chdir(BASE_DIR)
print(f"[cwd set] {BASE_DIR}")

nodes = [500,1000,1500]
r_list = [2,3,4]
seeds = range(200)
sparse = 0.0
model = 'VGAE'
num_epoch = 2
learning_rate = 0.01
base_data_dir = '../datasets/simulation/generator'
output_base_dir = '../synthetic/simulation/vgae-sample'

for n in nodes:
    for r in r_list:
        for seed in seeds:
            data_path = os.path.join(base_data_dir, f'n={n}_r={r}_sparse={sparse}', f'seed={seed}.pkl')
            output_dir = os.path.join(output_base_dir, f'n={n}_r={r}_sparse={sparse}', f'seed={seed}')
            os.makedirs(output_dir, exist_ok=True)
            command = [
                'python', 'train.py',
                '--model', model,
                '--data_path', data_path,
                '--output_dir', output_dir,
                '--input_dim', str(n),
                '--r', str(r),
                '--num_epoch', str(num_epoch),
                '--learning_rate', str(learning_rate)
            ]
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error:{e}.")