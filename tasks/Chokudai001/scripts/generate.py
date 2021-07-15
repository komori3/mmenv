import os
import argparse
import subprocess

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
DEFAULT_SEED_PATH = os.path.join(DEFAULT_DATA_PATH, 'seeds.txt')
DEFAULT_INPUT_PATH = os.path.join(DEFAULT_DATA_PATH, 'input')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', type=str, default=os.path.join('..', 'src', 'generator'))
    parser.add_argument('-s', '--seed-file', type=str, default=DEFAULT_SEED_PATH)
    parser.add_argument('-i', '--input-dir', type=str, default=DEFAULT_INPUT_PATH)
    
    args = parser.parse_args()

    assert os.path.exists(args.generator), f'generator file {args.seed_file} not found'
    assert os.path.exists(args.seed_file), f'seed file {args.seed_file} not found'
    
    if not os.path.exists(args.input_dir): os.makedirs(args.input_dir)

    with open(args.seed_file) as f:
        seeds = [int(seed) for seed in str(f.read()).split('\n') if seed != '']
    
    for seed in seeds:
        with open(os.path.join(args.input_dir, f'{seed}.in'), 'w') as f:
            subprocess.run([args.generator, '-s', str(seed)], stdout=f)