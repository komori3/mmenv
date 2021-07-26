import os
import argparse
import subprocess
from datetime import datetime
import shutil
from time import perf_counter_ns, sleep
import json
import yaml
from yaml.loader import SafeLoader
from multiprocessing import Pool


def get_path(config_file_path, relpath):
    return os.path.join(os.path.dirname(config_file_path), relpath)


def failed_with_status(seed: int, status: str) -> dict:
    print(f'seed = {seed}, score = -1, elapsed_ms = -1, status = {status}')
    return {'seed': seed, 'score': -1, 'elapsed_ms': -1, 'status': status}


def do_task(seed: int, generator_exec: str, solver_exec: str, judge_exec: str, input_dir: str, output_dir: str, error_dir: str, timelimit_ms: int) -> dict:
    # generate
    try:
        input_data = subprocess.check_output([generator_exec, '-s', str(seed)])
        input_file = os.path.join(input_dir, f'{seed}.in')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(input_data.decode(encoding='utf-8'))
    except Exception as e:
        print(e)
        return failed_with_status(seed, 'GE')  # generator error
    # exec
    output_file = os.path.join(output_dir, f'{seed}.out')
    error_file = os.path.join(error_dir, f'{seed}.err')
    try:
        with open(output_file, 'w', encoding='utf-8') as out:
            with open(error_file, 'w', encoding='utf-8') as err:
                elapsed = perf_counter_ns()
                subprocess.run([solver_exec], input=input_data,
                               stdout=out, stderr=err, timeout=timelimit_ms/1000.0)
                elapsed = perf_counter_ns() - elapsed
    except subprocess.TimeoutExpired as e:
        print(e)
        return failed_with_status(seed, 'TLE')  # time limit exceeded
    except Exception as e:
        print(e)
        return failed_with_status(seed, 'RE')  # runtime error
    # judge
    try:
        judge_cmd = [judge_exec, '-i', input_file, '-o', output_file]
        score = subprocess.check_output(judge_cmd).decode(encoding='utf-8')
        score = float(list(map(str.strip, score[:-1].split('=')))[1])
    except Exception as e:
        print(e)
        return failed_with_status(seed, 'JE')  # judge error

    print(
        f'seed = {seed}, score = {score}, elapsed_ms = {round(elapsed / 1000000.0, 1)}, status: AC')
    return {'seed': seed, 'score': score, 'elapsed_ms': round(elapsed / 1000000.0, 1), 'status': 'AC'}


def do_task_wrapper(param): return do_task(*param)


if __name__ == "__main__":
    timestamp = datetime.now()

    # TODO: 入力ファイル読み込み可能にする
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file path')
    parser.add_argument(
        '--tag', type=str, default=f's{timestamp.strftime("%Y%m%d%H%M%S")}', help='submission tag (name)')
    parser.add_argument('--test', type=int,
                        help='run example test for n test cases')
    parser.add_argument('-j', '--njobs', type=int, default=1,
                        help='multiprocess thread size')
    parser.add_argument('--vis', action='store_true',
                        help='include opencv files')

    args = parser.parse_args()

    # set params
    config_file = args.config
    assert os.path.exists(
        config_file), f'Config file {config_file} does not exist.'

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)

    submissions_dir = get_path(
        config_file, config['submissions_dir']) if args.test is None else 'example_test'
    submission_dir = os.path.join(submissions_dir, args.tag)
    assert not os.path.exists(
        submission_dir), f'Submission directory {submission_dir} already exists.'

    seed_file = get_path(config_file, config['seed_file'])
    assert os.path.exists(seed_file), f'Seed file {seed_file} does not exist.'

    generator_exec = get_path(config_file, config['generator'])
    assert os.path.exists(
        generator_exec), f'Generator program {generator_exec} does not exist.'

    judge_exec = get_path(config_file, config['judge'])
    assert os.path.exists(
        judge_exec), f'Judge program {judge_exec} does not exist.'

    source_file = get_path(config_file, config['source'])
    assert os.path.exists(
        judge_exec), f'Source file {source_file} does not exist.'

    if args.vis:
        assert args.test is not None, f'Visualize mode is only available if args.test is True.'

    # build
    try:
        solver_exec = './solver'
        build_cmd = ['g++', '-O2', '-Wall', '-Wextra',
                     '-std=c++17', '-o', solver_exec, source_file]
        if args.vis:
            build_cmd += [
                '-I/usr/local/include/opencv4', '-L/usr/local/lib',
                '-lopencv_core', '-lopencv_highgui', '-lopencv_imgproc'
            ]
        subprocess.run(build_cmd).check_returncode()
    except Exception as e:
        print(e)
        exit(1)

    # do tasks
    try:
        input_dir = os.path.join(submission_dir, 'in')
        output_dir = os.path.join(submission_dir, 'out')
        error_dir = os.path.join(submission_dir, 'err')
        os.makedirs(submission_dir)
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        os.makedirs(error_dir)
        shutil.copy2(source_file, submission_dir)

        with open(seed_file) as f:
            seeds = [int(seed)
                     for seed in str(f.read()).split('\n') if seed != '']
            if args.test is not None:
                num_seeds = len(seeds)
                seeds = seeds[:min(num_seeds, args.test)]

        summary = {}
        summary['submission_datetime'] = timestamp.strftime(
            "%Y-%m-%d %H:%M:%S")
        summary['tag'] = args.tag

        tasks = []
        for seed in seeds:
            tasks.append([seed, generator_exec, solver_exec, judge_exec,
                         input_dir, output_dir, error_dir, config['timelimit_ms']])

        pool = Pool(args.njobs)
        summary['results'] = pool.map(do_task_wrapper, tasks)

        score = 0
        for result in summary['results']:
            score += result['score']

        print(f'total score = {score}')
        summary['score'] = score

        summary_file = os.path.join(submission_dir, 'summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    except BaseException as e:
        print(e)
        sleep(1)
        shutil.rmtree(submission_dir)
        exit(1)

    if args.test is not None:
        shutil.rmtree(submissions_dir)

# python scripts/submit.py --config tasks/Chokudai001/config.yaml --tag test_submit -j 10
