import os
import sys
import argparse
import subprocess
from datetime import datetime
import shutil
from time import perf_counter_ns, sleep
import json
import yaml
from yaml.loader import SafeLoader
from multiprocessing import Pool


def failed_with_status(seed: int, status: str) -> dict:
    print(f'seed = {seed}, score = -1, elapsed_ms = -1, status = {status}')
    return {'seed': seed, 'score': -1, 'elapsed_ms': -1, 'status': status}


def do_task(seed: int, solver_exec: str, judge_exec: str, input_dir: str, output_dir: str, error_dir: str, timelimit_ms: int) -> dict:
    # read input
    try:
        input_file = os.path.join(input_dir, f'{seed}.in')
    except Exception as e:
        print(e)
        return failed_with_status(seed, 'GE')  # generator error
    # exec
    output_file = os.path.join(output_dir, f'{seed}.out')
    error_file = os.path.join(error_dir, f'{seed}.err')
    try:
        with open(input_file, 'r', encoding='utf-8') as input:
            with open(output_file, 'w', encoding='utf-8') as out:
                with open(error_file, 'w', encoding='utf-8') as err:
                    elapsed = perf_counter_ns()
                    subprocess.run([solver_exec], stdin=input,
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
        judge_cmd = ['python', judge_exec, input_file, output_file]
        score = subprocess.check_output(judge_cmd).decode(encoding='utf-8')
        score = float(list(map(str.strip, score[:-1].split(':')))[1])
    except Exception as e:
        print(e)
        return failed_with_status(seed, 'JE')  # judge error

    print(
        f'seed = {seed}, score = {score}, elapsed_ms = {round(elapsed / 1000000.0, 1)}, status: AC')
    return {'seed': seed, 'score': score, 'elapsed_ms': round(elapsed / 1000000.0, 1), 'status': 'AC'}


def do_task_wrapper(param): return do_task(*param)


class TaskRunnerBase:

    def get_path(self, relpath):
        return os.path.join(os.path.dirname(self.config_path), relpath)

    def __init__(self, args):

        self.timestamp = datetime.now()

        self.args = args

        self.config_path = args.config
        assert os.path.exists(
            self.config_path), f'Config file {self.config_path} does not exist.'
        self.config_path = os.path.abspath(self.config_path)

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=SafeLoader)

        self.submissions_dir = os.path.abspath(self.get_path(
            self.config['submissions_dir']) if args.test is None else 'example_test')

        self.submission_dir = os.path.join(self.submissions_dir, args.tag)
        assert not os.path.exists(
            self.submission_dir), f'Submission directory {self.submission_dir} already exists.'

        self.source_path = os.path.abspath(
            self.get_path(self.config['source']))
        assert os.path.exists(
            self.source_path), f'Source file {self.source_path} does not exist.'

        if args.vis:
            assert args.test is not None, f'Visualize mode is only available if args.test is True.'


class GeneralTaskRunner(TaskRunnerBase):

    def __init__(self, args):
        super().__init__(args)

        self.input_dir = self.get_path(self.config['input_dir'])
        assert os.path.exists(self.input_dir), f'Input directory {self.input_dir} does not exist.'

        self.seed_list_path = self.get_path(self.config['seed_list'])
        assert os.path.exists(
            self.seed_list_path), f'Seed list file {self.seed_list_path} does not exist.'

        self.judge_path = self.get_path(self.config['judge'])
        assert os.path.exists(
            self.judge_path), f'Judge program {self.judge_path} does not exist.'

    def build(self):

        self.solver_path = './solver'  # tmp

        build_cmd = ['g++-9', '-O2', '-Wall', '-Wextra',
                     '-std=c++17', '-o', self.solver_path, self.source_path]

        if self.args.vis:
            build_cmd += [
                '-I/usr/local/include/opencv4', '-L/usr/local/lib',
                '-lopencv_core', '-lopencv_highgui', '-lopencv_imgproc'
            ]

        subprocess.run(build_cmd).check_returncode()

    def do_tasks(self):

        try:

            input_dir = self.input_dir
            output_dir = os.path.join(self.submission_dir, 'out')
            error_dir = os.path.join(self.submission_dir, 'err')
            os.makedirs(self.submission_dir)
            os.makedirs(output_dir)
            os.makedirs(error_dir)
            shutil.copy2(self.source_path, self.submission_dir)

            with open(self.seed_list_path, 'r', encoding='utf-8') as f:
                seeds = [int(seed) for seed in str(
                    f.read()).split('\n') if seed != '']
                if self.args.test is not None:
                    num_seeds = len(seeds)
                    seeds = seeds[:min(num_seeds, self.args.test)]

            summary = {}
            summary['submission_datetime'] = self.timestamp.strftime(
                "%Y-%m-%d %H:%M:%S")
            summary['tag'] = self.args.tag
            summary['src'] = os.path.basename(self.source_path)

            tasks = []
            for seed in seeds:
                tasks.append([seed, self.solver_path, self.judge_path,
                              input_dir, output_dir, error_dir, self.config['timelimit_ms']])

            pool = Pool(self.args.njobs)
            summary['results'] = pool.map(do_task_wrapper, tasks)

            score = 0
            for result in summary['results']:
                score += result['score']

            print(f'total score = {score}')
            summary['score'] = score

            summary_path = os.path.join(
                self.submission_dir, 'summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)

        except BaseException as e:

            print(e)
            sleep(1)
            shutil.rmtree(self.submission_dir)
            if self.args.test is not None:
                shutil.rmtree(self.submissions_dir)
            exit(1)

        if self.args.test is not None:
            shutil.rmtree(self.submissions_dir)


class TCTaskRunner(TaskRunnerBase):

    def __init__(self, args):
        super().__init__(args)

        self.seed_range = self.config[
            'seed_range'] if self.args.test is None else f'1+{self.args.test}'

        self.tester_path = self.get_path(self.config['tester'])

    def build(self):

        self.solver_path = os.path.abspath(os.path.join('solver'))  # tmp

        self.build_cmd = ['g++-7', '-O3', '-Wall', '-Wextra',
                          '-std=gnu++11', '-o', self.solver_path, self.source_path]

        if self.args.vis:
            self.build_cmd += [
                '-I/usr/local/include/opencv4', '-L/usr/local/lib',
                '-lopencv_core', '-lopencv_highgui', '-lopencv_imgproc'
            ]

        subprocess.run(self.build_cmd).check_returncode()

    def parse_result(self, line):
        result = {}
        cols = line.split(',')
        for col in cols:
            key, val = tuple(map(str.strip, col.split('=')))
            if key == 'Seed':
                val = int(val)
            elif key == 'Score':
                val = float(val)
            result[key.lower()] = val
        result['status'] = 'WA' if result['score'] < - \
            0.5 else 'AC'  # TODO: TLE
        return result

    def do_tasks(self):
        # tester directory で tester を走らせないといけない

        try:

            self.input_dir = os.path.join(self.submission_dir, 'in')
            self.output_dir = os.path.join(self.submission_dir, 'out')
            self.error_dir = os.path.join(self.submission_dir, 'err')
            os.makedirs(self.submission_dir)
            shutil.copy2(self.source_path, self.submission_dir)

            tester_dir = os.path.dirname(self.tester_path)
            tester_name = os.path.basename(self.tester_path)

            os.chdir(tester_dir)

            options = ['-nv', '-no', '-pr', f'-si {self.input_dir}',
                       f'-so {self.output_dir}', f'-se {self.error_dir}']
            options.append(f'-th {self.args.njobs}')
            options.append(f'-tl {self.config["timelimit_ms"]}')

            tester_cmd = f'java -jar {tester_name} -ex {self.solver_path} -sd {self.seed_range} {" ".join(options)}'

            proc = subprocess.Popen(
                tester_cmd, shell=True, stdout=subprocess.PIPE)
            results = []
            while True:
                line = proc.stdout.readline().decode()[:-1]
                if len(line) > 0 and line[0] == 'S':
                    result = self.parse_result(line)
                    results.append(result)
                sys.stdout.write(line + '\n')
                if not line and proc.poll() is not None:
                    break

            results = sorted(results, key=lambda x: x['seed'])

            summary = {}
            summary['submission_datetime'] = self.timestamp.strftime(
                "%Y-%m-%d %H:%M:%S")
            summary['tag'] = self.args.tag
            summary['src'] = os.path.basename(self.source_path)

            summary['results'] = results

            summary_path = os.path.join(self.submission_dir, 'summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)

        except BaseException as e:

            print(e)
            sleep(1)
            shutil.rmtree(self.submission_dir)
            if self.args.test is not None:
                shutil.rmtree(self.submissions_dir)

        if self.args.test is not None:
            shutil.rmtree(self.submissions_dir)


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

    try:
        task_runner = GeneralTaskRunner(args)
        task_runner.build()
        task_runner.do_tasks()

    except Exception as e:

        print(e)
        exit(1)

# python scripts/submit.py --config tasks/Chokudai001/config.yaml --tag test_submit -j 10
