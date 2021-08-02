import os
import inspect
from datetime import datetime
import argparse
import yaml
from yaml.loader import SafeLoader
import json


def location(depth=0):
    frame = inspect.currentframe().f_back
    return os.path.basename(frame.f_code.co_filename), frame.f_code.co_name, frame.f_lineno


_registry = {}


def register(*alias_names):
    def _decorator(f):
        def _factory(*vargs):
            return f(*vargs)
        for key in alias_names:
            _registry[key] = _factory
        return _factory
    return _decorator


def get_path(config_path, relpath):
    return os.path.join(os.path.dirname(config_path), relpath)


def set_score_sum(data: dict):
    dict_submission_to_total_score = {}
    for tag, summary in data.items():
        results = summary['results']
        dict_submission_to_total_score[tag] = 0.0
        for result in results:
            score = float(result['score'])
            if result['status'] == 'AC':
                dict_submission_to_total_score[tag] += score
    for tag, summary in data.items():
        summary['score'] = dict_submission_to_total_score[tag]


@register('greater')
def greater(data: dict):
    set_score_sum(data)
    return sorted(list(data.values()), key=lambda x: x['score'], reverse=True)


@register('less')
def less(data: dict):
    set_score_sum(data)
    return sorted(list(data.values()), key=lambda x: x['score'])


@register('yours_div_max')
def yours_div_max(data: dict):
    dict_submission_to_scores = {}
    dict_submission_to_total_score = {}
    seed_set = set()
    for tag, summary in data.items():
        dict_submission_to_scores[tag] = {}
        dict_submission_to_total_score[tag] = 0.0
        for result in summary['results']:
            seed = int(result['seed'])
            score = float(result['score'])
            seed_set.add(seed)
            dict_submission_to_scores[tag][seed] = score if result['status'] == 'AC' else 0.0
    for seed in seed_set:
        max_score = 0.0
        dict_submission_to_seed_score = {}
        for tag, submission_scores in dict_submission_to_scores.items():
            dict_submission_to_seed_score[tag] = submission_scores[seed]
            max_score = max(max_score, submission_scores[seed])
        if max_score < 1e-8:
            continue
        for tag, score in dict_submission_to_seed_score.items():
            dict_submission_to_total_score[tag] += score / max_score
    for tag, summary in data.items():
        summary['score'] = dict_submission_to_total_score[tag]
    print(dict_submission_to_total_score)
    return sorted(list(data.values()), key=lambda x: x['score'], reverse=True)


@register('min_div_yours')
def min_div_yours(data: dict):
    dict_submission_to_scores = {}
    dict_submission_to_total_score = {}
    seed_set = set()
    for tag, summary in data.items():
        dict_submission_to_scores[tag] = {}
        dict_submission_to_total_score[tag] = 0.0
        for result in summary['results']:
            seed = int(result['seed'])
            score = float(result['score'])
            seed_set.add(seed)
            dict_submission_to_scores[tag][seed] = score if result['status'] == 'AC' else None
    for seed in seed_set:
        min_score = float('inf')
        dict_submission_to_seed_score = {}
        for tag, submission_scores in dict_submission_to_scores.items():
            dict_submission_to_seed_score[tag] = submission_scores[seed]
            if submission_scores[seed] is None:
                continue
            min_score = min(min_score, submission_scores[seed])
        if min_score == float('inf'):
            continue
        for tag, score in dict_submission_to_seed_score.items():
            if score is None:
                continue
            dict_submission_to_total_score[tag] += min_score / score
    for tag, summary in data.items():
        summary['score'] = dict_submission_to_total_score[tag]
    return sorted(list(data.values()), key=lambda x: x['score'], reverse=True)


def extract_data(submissions_dir):
    data = {}
    for tag in os.listdir(submissions_dir):
        submission_dir = os.path.join(submissions_dir, tag)
        summary_path = os.path.join(submission_dir, 'summary.json')
        with open(summary_path, 'r', encoding='utf-8') as f:
            data[tag] = json.load(f)
    return data


def create_evaluator(name):
    if name in _registry:
        return _registry[name]
    print(f'metric {name} is not registered.')
    raise RuntimeError


def show_standings(table: list):
    max_length = len('submission')
    list_total_score_to_submission = []
    for elem in table:
        tag = elem['tag']
        score = elem['score']
        max_length = max(max_length, len(tag))
        list_total_score_to_submission.append((score, tag))

    space = max_length - len('submission') + 4
    print('submission' + (' ' * space) + 'score')
    print('-' * 50)
    for score, tag in list_total_score_to_submission:
        space = max_length - len(tag) + 4
        print(tag + (' ' * space) + str(score))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file path')

    args = parser.parse_args()

    config_path = args.config
    assert os.path.exists(
        config_path), f'Config file {config_path} does not exist.'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)

    submissions_dir = get_path(config_path, config['submissions_dir'])

    data = extract_data(submissions_dir)

    evaluator = create_evaluator(config['metric'])

    table = evaluator(data)

    show_standings(table)


# python scripts/evaluate.py --config tasks/Chokudai001/config.yaml
