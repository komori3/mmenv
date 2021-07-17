import os
import argparse
import subprocess
from datetime import datetime
import shutil
from time import perf_counter_ns
import json
import yaml
from yaml.loader import SafeLoader

def get_path(config_file_path, relpath):
    return os.path.join(os.path.dirname(config_file_path), relpath)

if __name__ == "__main__":
    timestamp = datetime.now()

    # TODO: 入力ファイル読み込み可能にする
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--tag', type=str, default=f's{timestamp.strftime("%Y%m%d%H%M%S")}', help='submission tag (name)')
    parser.add_argument('--test', type=int, help='run example test for n test cases')
    
    args = parser.parse_args()

    config_file = args.config
    assert os.path.exists(config_file), f'Config file {config_file} does not exist.'

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)

    submissions_dir = get_path(config_file, config['submissions_dir']) if args.test is None else 'example_test'
    submission_dir = os.path.join(submissions_dir, args.tag)
    assert not os.path.exists(submission_dir), f'Submission directory {submission_dir} already exists.'

    seed_file = get_path(config_file, config['seed_file'])
    assert os.path.exists(seed_file), f'Seed file {seed_file} does not exist.'

    generator_exec = get_path(config_file, config['generator'])
    assert os.path.exists(generator_exec), f'Generator program {generator_exec} does not exist.'

    judge_exec = get_path(config_file, config['judge'])
    assert os.path.exists(judge_exec), f'Judge program {judge_exec} does not exist.'

    source_file = get_path(config_file, config['source'])
    assert os.path.exists(judge_exec), f'Source file {source_file} does not exist.'

    os.makedirs(submission_dir)

    solver_exec = './solver'
    build_cmd = ['g++', '-O2', '-Wall', '-Wextra', '-std=c++17', '-o', solver_exec, source_file]
    subprocess.run(build_cmd).check_returncode()

    input_dir = os.path.join(submission_dir, 'in')
    output_dir = os.path.join(submission_dir, 'out')
    result_dir = os.path.join(submission_dir, 'result')
    os.makedirs(input_dir)
    os.makedirs(output_dir)
    os.makedirs(result_dir)

    shutil.copy2(source_file, submission_dir)

    with open(seed_file) as f:
        seeds = [int(seed) for seed in str(f.read()).split('\n') if seed != '']
        if args.test is not None:
            num_seeds = len(seeds)
            seeds = seeds[:min(num_seeds, args.test)]
    
    meta_info = {}
    meta_info['submission_datetime'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    meta_info['tag'] = args.tag
    meta_info['results'] = []
    for seed in seeds:
        input_data = subprocess.check_output([generator_exec, '-s', str(seed)])
        input_file = os.path.join(input_dir, f'{seed}.in')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(input_data.decode(encoding='utf-8'))
        elapsed = perf_counter_ns()
        output_data = subprocess.check_output([solver_exec], input=input_data)
        elapsed = perf_counter_ns() - elapsed
        # TODO: timelimit_ms 対応
        output_file = os.path.join(output_dir, f'{seed}.out')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_data.decode(encoding='utf-8'))
        result_file = os.path.join(result_dir, f'{seed}.result')
        judge_cmd = [judge_exec, '-i', input_file, '-o', output_file, '-r', result_file]
        score = subprocess.check_output(judge_cmd).decode(encoding='utf-8')
        score = float(list(map(str.strip, score[:-1].split('=')))[1])
        print(f'seed = {seed}, score = {score}, elapsed_ms = {round(elapsed / 1000000.0, 1)}')
        meta_info['results'].append({'seed': seed, 'score': score, 'elapsed_ms': round(elapsed / 1000000.0, 1)})

    meta_file = os.path.join(submission_dir, 'meta.json')
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2)

    if args.test is not None:
        shutil.rmtree(submissions_dir)

# python scripts/submit_by_config.py --config tasks/Chokudai001/config.yaml --tag test_submit
