import os
import argparse
import subprocess
from datetime import datetime
import shutil

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
DEFAULT_SEED_PATH = os.path.join(DEFAULT_DATA_PATH, 'seeds.txt')
DEFAULT_INPUT_PATH = os.path.join(DEFAULT_DATA_PATH, 'input')
DEFAULT_SUBMISSION_PATH = os.path.join(DEFAULT_DATA_PATH, 'submissions', f's{datetime.now().strftime("%Y%m%d%H%M%S")}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--judge', type=str, default=os.path.join('..', 'src', 'judge'))
    parser.add_argument('-s', '--seed-file', type=str, default=DEFAULT_SEED_PATH)
    parser.add_argument('-i', '--input-dir', type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument('-o', '--output-dir', type=str, default=DEFAULT_SUBMISSION_PATH)
    
    args = parser.parse_args()

    assert os.path.exists(args.source), f'source file {args.source} not found'
    assert os.path.exists(args.judge), f'judge file {args.judge} not found'
    assert os.path.exists(args.seed_file), f'seed file {args.seed_file} not found'
    assert os.path.exists(args.input_dir), f'directory {args.inupt_dir} not found'
    assert not os.path.exists(args.output_dir), f'submission directory {args.output_dir} already exists'

    exec_cmd = ['g++', '-O2', '-Wall', '-Wextra', '-std=c++17', args.source]
    subprocess.run(exec_cmd).check_returncode()

    input_dir = args.input_dir
    submission_dir = args.output_dir
    output_dir = os.path.join(submission_dir, 'output')
    result_dir = os.path.join(submission_dir, 'result')
    os.makedirs(submission_dir)
    os.makedirs(output_dir)
    os.makedirs(result_dir)

    shutil.copy2(args.source, os.path.join(submission_dir, os.path.basename(args.source)))

    with open(args.seed_file) as f:
        seeds = [int(seed) for seed in str(f.read()).split('\n') if seed != '']

    for seed in seeds:
        input_file_name = os.path.join(input_dir, f'{seed}.in')
        output_file_name = os.path.join(output_dir, f'{seed}.out')
        result_file_name = os.path.join(result_dir, f'{seed}.result')
        with open(input_file_name) as input_file:
            with open(output_file_name, 'w') as output_file:
                subprocess.run('./a.out', stdin=input_file, stdout=output_file)
        subprocess.run([args.judge, '-i', input_file_name, '-o', output_file_name, '-r', result_file_name])