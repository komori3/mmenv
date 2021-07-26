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
from submit import get_path, do_task_wrapper

if __name__ == "__main__":
    timestamp = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file path')
    parser.add_argument(
        '--tag', type=str, help='submission tag (name)')  # None なら全部リジャッジ
    parser.add_argument('-j', '--njobs', type=int, default=1,
                        help='multiprocess thread size')

    args = parser.parse_args()

    # set params
    config_file = args.config
    assert os.path.exists(
        config_file), f'Config file {config_file} does not exist.'

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)

    base_submissions_dir = get_path(config_file, config['submissions_dir'])
    submission_tags = os.listdir(
        base_submissions_dir) if args.tag is None else [args.tag]

    for tag in submission_tags:
        base_submission_dir = os.path.join(base_submissions_dir, tag)
        assert os.path.exists(
            base_submission_dir), f'Submission directory {base_submission_dir} does not exist.'

    seed_file = get_path(config_file, config['seed_file'])
    assert os.path.exists(seed_file), f'Seed file {seed_file} does not exist.'

    generator_exec = get_path(config_file, config['generator'])
    assert os.path.exists(
        generator_exec), f'Generator program {generator_exec} does not exist.'

    judge_exec = get_path(config_file, config['judge'])
    assert os.path.exists(
        judge_exec), f'Judge program {judge_exec} does not exist.'

    dest_submissions_dir = os.path.join(os.path.dirname(
        base_submissions_dir), f'rejudge{timestamp.strftime("%Y%m%d%H%M%S")}')

    try:

        for tag in submission_tags:

            base_submission_dir = os.path.join(base_submissions_dir, tag)
            dest_submission_dir = os.path.join(dest_submissions_dir, tag)
            os.makedirs(dest_submission_dir)
            base_summary_file = os.path.join(
                base_submission_dir, 'summary.json')
            with open(base_summary_file, 'r', encoding='utf-8') as f:
                base_summary = json.load(f)
            base_src_file = os.path.join(
                base_submission_dir, base_summary['src'])
            assert os.path.exists(base_src_file)
            dest_src_file = os.path.join(
                dest_submission_dir, base_summary['src'])
            shutil.copy2(base_src_file, dest_src_file)

            # build
            solver_exec = './solver'
            build_cmd = ['g++', '-O2', '-Wall', '-Wextra',
                         '-std=c++17', '-o', solver_exec, base_src_file]
            subprocess.run(build_cmd).check_returncode()

            # do tasks
            input_dir = os.path.join(dest_submission_dir, 'in')
            output_dir = os.path.join(dest_submission_dir, 'out')
            error_dir = os.path.join(dest_submission_dir, 'err')
            os.makedirs(input_dir)
            os.makedirs(output_dir)
            os.makedirs(error_dir)

            with open(seed_file) as f:
                seeds = [int(seed)
                         for seed in str(f.read()).split('\n') if seed != '']

            dest_summary = {}
            dest_summary['submission_datetime'] = timestamp.strftime(
                "%Y-%m-%d %H:%M:%S")
            dest_summary['tag'] = base_summary['tag']
            dest_summary['src'] = base_summary['src']

            tasks = []
            for seed in seeds:
                tasks.append([seed, generator_exec, solver_exec, judge_exec,
                              input_dir, output_dir, error_dir, config['timelimit_ms']])

            pool = Pool(args.njobs)
            dest_summary['results'] = pool.map(do_task_wrapper, tasks)

            score = 0
            for result in dest_summary['results']:
                score += result['score']

            print(f'total score = {score}')
            dest_summary['score'] = score

            dest_summary_file = os.path.join(
                dest_submission_dir, 'summary.json')

            with open(dest_summary_file, 'w', encoding='utf-8') as f:
                json.dump(dest_summary, f, indent=2)

    except BaseException as e:
        print(e)
        sleep(1)
        shutil.rmtree(dest_submissions_dir)

    for tag in submission_tags:
        base_submission_dir = os.path.join(base_submissions_dir, tag)
        dest_submission_dir = os.path.join(dest_submissions_dir, tag)
        shutil.rmtree(base_submission_dir)
        shutil.copytree(dest_submission_dir, base_submission_dir)

    shutil.rmtree(dest_submissions_dir)

# python scripts/rejudge.py --config tasks/Chokudai001/config.yaml -j 10
# python scripts/rejudge.py --config tasks/Chokudai001/config.yaml -j 10 --tag trivial