import os
import argparse
import subprocess
from datetime import datetime
import shutil
from time import perf_counter_ns
import json

if __name__ == "__main__":
    # TODO: 入力ファイル読み込み可能にする
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='source file')
    parser.add_argument('--gen', type=str, required=True, help='generator program')
    parser.add_argument('--jdg', type=str, required=True, help='judge program')
    parser.add_argument('--dir', type=str, required=True, help='submission directory')
    parser.add_argument('--sd', type=str, required=True, help='seed file')
    
    args = parser.parse_args()

    assert os.path.exists(args.src), f'Source file {args.source} does not exist.'
    assert os.path.exists(args.gen), f'Generator program {args.gen} does not exist.'
    assert os.path.exists(args.jdg), f'Judge program {args.gen} does not exist.'
    assert not os.path.exists(args.dir), f'Submission directory {args.dir} already exists.'
    assert os.path.exists(args.sd), f'Seed file {args.sd} does not exist.'

    solver_exec = './solver'
    build_cmd = ['g++', '-O2', '-Wall', '-Wextra', '-std=c++17', '-o', solver_exec, args.src]
    subprocess.run(build_cmd).check_returncode()

    input_dir = os.path.join(args.dir, 'in')
    output_dir = os.path.join(args.dir, 'out')
    result_dir = os.path.join(args.dir, 'result')
    os.makedirs(input_dir)
    os.makedirs(output_dir)
    os.makedirs(result_dir)

    shutil.copy2(args.src, args.dir)

    with open(args.sd) as f:
        seeds = [int(seed) for seed in str(f.read()).split('\n') if seed != '']
    
    meta_info = {}
    meta_info['submission_datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_info['results'] = []
    for seed in seeds:
        input_data = subprocess.check_output([args.gen, '-s', str(seed)])
        input_file = os.path.join(input_dir, f'{seed}.in')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(input_data.decode(encoding='utf-8'))
        elapsed = perf_counter_ns()
        output_data = subprocess.check_output([solver_exec], input=input_data)
        elapsed = perf_counter_ns() - elapsed
        output_file = os.path.join(output_dir, f'{seed}.out')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_data.decode(encoding='utf-8'))
        result_file = os.path.join(result_dir, f'{seed}.result')
        judge_cmd = [args.jdg, '-i', input_file, '-o', output_file, '-r', result_file]
        score = subprocess.check_output(judge_cmd).decode(encoding='utf-8')
        score = float(list(map(str.strip, score[:-1].split('=')))[1])
        print(f'seed = {seed}, score = {score}, elapsed_ms = {round(elapsed / 1000000.0, 1)}')
        meta_info['results'].append({'seed': seed, 'score': score, 'elapsed_ms': round(elapsed / 1000000.0, 1)})

    meta_file = os.path.join(args.dir, 'meta.json')
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2)
    
# python submit2.py --src ../src/solvers/solver.cpp --gen ../src/generator --jdg ../src/judge --sd ../data/seeds.txt --dir ../data/submissions/test
