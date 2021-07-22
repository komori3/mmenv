#!/usr/bin/python3

from typing import Awaitable
import flask
from flask import request
import os
import json
import yaml
from yaml.loader import SafeLoader
from functools import cmp_to_key

app = flask.Blueprint('tasks', __name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(ROOT_DIR, 'tasks')
WWW_DIR = os.path.join(ROOT_DIR, 'www')



def load_task_config_yaml(task_tag: str):
    config_path = os.path.join(TASKS_DIR, task_tag, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config

def parse_task(task_tag: str):
    task = load_task_config_yaml(task_tag)
    return task

def struct_tasks():
    tasks = [parse_task(tag) for tag in os.listdir(TASKS_DIR)]
    return tasks

@app.route('/tasks')
def show_tasks():
    context = {
        'tasks': struct_tasks(),
        'active_tasks': 'active'
    }
    return flask.render_template('tasks.html', title='Tasks', **context)



def get_path(config_file_path, relpath):
    return os.path.join(os.path.dirname(config_file_path), relpath)

def parse_submission(submissions_dir, submission_tag):
    submission_dir = os.path.join(submissions_dir, submission_tag)
    summary_file = os.path.join(submission_dir, 'summary.json')
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    return summary

def set_bgcolor(row: list):
    max_score = -1e9
    min_score = 1e9
    for col in range(1, len(row)):
        if 'value' in row[col]:
            max_score = max(max_score, row[col]['value'])
            min_score = min(min_score, row[col]['value'])
    col_white = [255, 255, 255]
    col_green = [195, 230, 203]
    col_red = [230, 195, 203]
    def rgb(r, g, b):
        return '#' + hex(round(r))[2:] + hex(round(g))[2:] + hex(round(b))[2:]
    def arrmul(arr: list, val: float):
        return list(map(lambda x: x * val, arr))
    def arrsum(arr1: list, arr2: list):
        return list(map(lambda x,y: x + y, arr1, arr2))
    def get_color(ratio: float):
        return rgb(*arrsum(arrmul(col_green, ratio), arrmul(col_white, 1.0 - ratio)))
    for col in range(1, len(row)):
        if 'value' in row[col]:
            row[col]['bgcolor'] = get_color((row[col]['value'] - min_score) / (max_score - min_score))
        else:
            row[col]['bgcolor'] = rgb(*col_red)
    return row

def set_link(header: list, rows: dict):
    for seed, row in rows.items():
        if type(seed) is not int: continue
        for j in range(1, len(header)):
            if 'value' in row[j]:
                row[j]['link'] = os.path.join(header[j], str(seed))

def struct_submissions(submissions_dir: str):
    submissions = [parse_submission(submissions_dir, submission) for submission in os.listdir(submissions_dir)]
    submissions.sort(key=lambda x: x['score'], reverse=True)
    ncols = len(submissions) + 1
    header = ['seed']

    def set_row(rows, name):
        rows[name] = []
        for _ in range(ncols): rows[name].append({})

    rows = {}
    set_row(rows, 'all')
    rows['all'][0]['value'] = 'all'
    for col, submission in enumerate(submissions):
        col += 1
        tag = submission['tag']
        header += [tag]
        for result in submission['results']:
            seed = result['seed']
            if not seed in rows:
                set_row(rows, seed)
                rows[seed][0]['value'] = seed
            rows[seed][col]['value'] = result['score']
        rows['all'][col]['value'] = submission['score']
    
    set_link(header, rows)

    rows = [set_bgcolor(row) for _, row in rows.items()]

    return {'header': header, 'rows': rows}

@app.route('/tasks/<tag>')
def show_task(tag):

    config_path = os.path.join(TASKS_DIR, tag, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)

    submissions_dir = get_path(config_path, config['submissions_dir'])

    context = {
        'task': config,
        'submissions': struct_submissions(submissions_dir)
    }

    return flask.render_template('task.html', title=tag, **context)

@app.route('/tasks/<tag>/<sol>/<seed>')
def show_result(tag, sol, seed):

    config_path = os.path.join(TASKS_DIR, tag, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)

    submissions_dir = get_path(config_path, config['submissions_dir'])
    submission_dir = os.path.join(submissions_dir, sol)
    input_file = os.path.join(submission_dir, 'in', f'{seed}.in')
    output_file = os.path.join(submission_dir, 'out', f'{seed}.out')

    with open(input_file, 'r', encoding='utf-8') as f:
        input_txt = str(f.read())
    with open(output_file, 'r', encoding='utf-8') as f:
        output_txt = str(f.read())

    context = {
        'tag': tag,
        'name': config['name'],
        'sol': sol,
        'seed': seed,
        'input_txt': input_txt,
        'output_txt': output_txt
    }

    print(tag, sol, seed)
    return flask.render_template(f'vis/{tag}.html', title=tag, **context)