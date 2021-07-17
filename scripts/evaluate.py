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

def show_standings(dict_submission_to_total_score):
    max_length = len('submission')
    list_total_score_to_submission = []
    for submission, total_score in dict_submission_to_total_score.items():
        max_length = max(max_length, len(submission))
        list_total_score_to_submission.append((total_score, submission))

    list_total_score_to_submission.sort()

    space = max_length - len('submission') + 4
    print('submission' + (' ' * space) + 'score')
    print('-' * 50)
    for total_score, submission in list_total_score_to_submission:
        space = max_length - len(submission) + 4
        print(submission + (' ' * space) + str(total_score))

if __name__ == "__main__":
    timestamp = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file path')
    
    args = parser.parse_args()

    config_file = args.config
    assert os.path.exists(config_file), f'Config file {config_file} does not exist.'

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)

    submissions_dir = get_path(config_file, config['submissions_dir'])
    assert os.path.exists(submissions_dir), f'Submissions directory {submissions_dir} does not exist.'

    # TODO: seed 部分集合指定可能にする / 特定パラメータ限定可能にする
    seed_file = get_path(config_file, config['seed_file'])
    assert os.path.exists(seed_file), f'Seed file {seed_file} does not exist.'

    with open(seed_file) as f:
        seeds = [int(seed) for seed in str(f.read()).split('\n') if seed != '']

    # TODO: metric 対応
    submission_to_score = {}
    for submission_dir_name in os.listdir(submissions_dir):
        submission_dir = os.path.join(submissions_dir, submission_dir_name)
        meta_file = os.path.join(submission_dir, 'meta.json')
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta_info = json.load(f)
        seed_to_score = {}
        for result in meta_info['results']:
            seed_to_score[result['seed']] = result['score']
        score = 0
        for seed in seeds:
            if seed in seed_to_score:
                score += seed_to_score[seed]
        submission_to_score[meta_info['tag']] = score

    show_standings(submission_to_score)

