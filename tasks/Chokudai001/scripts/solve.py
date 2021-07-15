import os
import argparse
import subprocess

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
DEFAULT_SEED_PATH = os.path.join(DEFAULT_DATA_PATH, 'seeds.txt')
DEFAULT_INPUT_PATH = os.path.join(DEFAULT_DATA_PATH, 'input')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, required=True)
    parser.add_argument('-s', '--seed-file', type=str, default=DEFAULT_SEED_PATH)
    parser.add_argument('-i', '--input-dir', type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    
    args = parser.parse_args()

    assert os.path.exists(args.solver), f'solver file {args.solver} not found'
    assert os.path.exists(args.seed_file), f'seed file {args.seed_file} not found'
    assert os.path.exists(args.input_dir), f'input directory {args.input_dir} not found'
    
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    with open(args.seed_file) as f:
        seeds = [int(seed) for seed in str(f.read()).split('\n') if seed != '']

    input_dir = args.input_dir
    output_dir = args.output_dir
    for seed in seeds:
        input_file_name = os.path.join(input_dir, f'{seed}.in')
        output_file_name = os.path.join(output_dir, f'{seed}.out')
        with open(input_file_name) as input_file:
            with open(output_file_name, 'w') as output_file:
                subprocess.run([args.solver], stdin=input_file, stdout=output_file)