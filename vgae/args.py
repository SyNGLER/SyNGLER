import argparse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--r', type=int, required=True)

    p.add_argument('--num_epoch', type=int, default=200)
    p.add_argument('--learning_rate', type=float, default=1e-2)

    p.add_argument('--use_feature', action='store_true',default = False, help='use provided node features X')
    p.add_argument('--input_dim', type=int, required=True)
    p.add_argument('--hidden1_dim', type=int, default=32)
    p.add_argument('--hidden2_dim', type=int, default=16)

    args = p.parse_args()
    return args

if __name__ == "__main__":
    _ = get_args()