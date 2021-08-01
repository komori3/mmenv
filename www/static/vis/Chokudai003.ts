declare var GIF: any;  // for https://github.com/jnordberg/gif.js

module framework {
    export class FileParser {
        private filename: string;
        private content: string[][];
        private y: number;
        private x: number

        constructor(filename: string, content: string) {
            this.filename = filename;
            this.content = [];
            for (const line of content.split('\n')) {
                if (line.length == 0) continue;
                const words = line.trim().split(new RegExp('\\s+'));
                this.content.push(words);
            }
            this.y = 0;
            this.x = 0;
        }

        public getWord(): string {
            if (this.content.length <= this.y) {
                this.reportError('a word expected, but EOF');
            }
            if (this.content[this.y].length <= this.x) {
                this.reportError('a word expected, but newline');
            }
            const word = this.content[this.y][this.x];
            this.x += 1;
            return word;
        }
        public getInt(): number {
            const word = this.getWord();
            if (!word.match(new RegExp('^[-+]?[0-9]+$'))) {
                this.reportError(`a number expected, but word ${JSON.stringify(this.content[this.y][this.x])}`);
            }
            return parseInt(word);
        }
        public getNewline() {
            if (this.content.length <= this.y) {
                this.reportError('newline expected, but EOF');
            }
            if (this.x < this.content[this.y].length) {
                this.reportError(`newline expected, but word ${JSON.stringify(this.content[this.y][this.x])}`);
            }
            this.x = 0;
            this.y += 1;
        }
        public isNewline(): boolean {
            return this.content[this.y].length <= this.x;
        }
        public isEOF(): boolean {
            return this.content.length <= this.y;
        }

        public unwind() {
            if (this.x == 0) {
                this.y -= 1;
                this.x = this.content[this.y].length - 1;
            } else {
                this.x -= 1;
            }
        }
        public reportError(msg: string) {
            msg = `${this.filename}: line ${this.y + 1}: ${msg}`;
            alert(msg);
            throw new Error(msg);
        }
    }

    export class FileSelector {
        public callback: (inputContent: string, outputContent: string) => void;

        private inputText: HTMLInputElement;
        private outputText: HTMLInputElement;
        private reloadButton: HTMLInputElement;

        constructor() {
            this.inputText = <HTMLInputElement>document.getElementById("inputText")
            this.outputText = <HTMLInputElement>document.getElementById("outputText");
            this.reloadButton = <HTMLInputElement>document.getElementById("reloadButton");
            this.reloadFilesClosure = () => { this.reloadFiles(); };
            this.inputText.addEventListener('change', this.reloadFilesClosure);
            this.outputText.addEventListener('change', this.reloadFilesClosure);
            this.reloadButton.addEventListener('click', this.reloadFilesClosure);
        }

        public clickReloadButton() {
            this.reloadButton.dispatchEvent(new Event('click'));
        }

        private reloadFilesClosure: () => void;
        reloadFiles() {
            if (this.inputText.value.length == 0 || this.outputText.value.length == 0) return;
            loadFile(this.inputText, (inputContent: string) => {
                loadFile(this.outputText, (outputContent: string) => {
                    this.inputText.removeEventListener('change', this.reloadFilesClosure);
                    this.outputText.removeEventListener('change', this.reloadFilesClosure);
                    this.reloadButton.classList.remove('disabled');
                    if (this.callback !== undefined) {
                        this.callback(inputContent, outputContent);
                    }
                });
            });
        }
    }

    export class RichSeekBar {
        public callback: (value: number) => void;

        private seekRange: HTMLInputElement;
        private seekNumber: HTMLInputElement;
        private fpsInput: HTMLInputElement;
        private firstButton: HTMLInputElement;
        private prevButton: HTMLInputElement;
        private playButton: HTMLInputElement;
        private nextButton: HTMLInputElement;
        private lastButton: HTMLInputElement;
        private runIcon: HTMLElement;
        private intervalId: number;
        private playClosure: () => void;
        private stopClosure: () => void;

        constructor() {
            this.seekRange = <HTMLInputElement>document.getElementById("seekRange");
            this.seekNumber = <HTMLInputElement>document.getElementById("seekNumber");
            this.fpsInput = <HTMLInputElement>document.getElementById("fpsInput");
            this.firstButton = <HTMLInputElement>document.getElementById("firstButton");
            this.prevButton = <HTMLInputElement>document.getElementById("prevButton");
            this.playButton = <HTMLInputElement>document.getElementById("playButton");
            this.nextButton = <HTMLInputElement>document.getElementById("nextButton");
            this.lastButton = <HTMLInputElement>document.getElementById("lastButton");
            this.runIcon = document.getElementById("runIcon");
            this.intervalId = null;

            this.setMinMax(-1, -1);
            this.seekRange.addEventListener('change', () => { this.setValue(parseInt(this.seekRange.value)); });
            this.seekNumber.addEventListener('change', () => { this.setValue(parseInt(this.seekNumber.value)); });
            this.seekRange.addEventListener('input', () => { this.setValue(parseInt(this.seekRange.value)); });
            this.seekNumber.addEventListener('input', () => { this.setValue(parseInt(this.seekNumber.value)); });
            this.fpsInput.addEventListener('change', () => { if (this.intervalId !== null) { this.play(); } });
            this.firstButton.addEventListener('click', () => { this.stop(); this.setValue(this.getMin()); });
            this.prevButton.addEventListener('click', () => { this.stop(); this.setValue(this.getValue() - 1); });
            this.nextButton.addEventListener('click', () => { this.stop(); this.setValue(this.getValue() + 1); });
            this.lastButton.addEventListener('click', () => { this.stop(); this.setValue(this.getMax()); });
            this.playClosure = () => { this.play(); };
            this.stopClosure = () => { this.stop(); };
            this.playButton.addEventListener('click', this.playClosure);
        }

        public setMinMax(min: number, max: number) {
            this.seekRange.min = this.seekNumber.min = min.toString();
            this.seekRange.max = this.seekNumber.max = max.toString();
            this.seekRange.step = this.seekNumber.step = '1';
            this.setValue(min);
        }
        public getMin(): number {
            return parseInt(this.seekRange.min);
        }
        public getMax(): number {
            return parseInt(this.seekRange.max);
        }

        public setValue(value: number) {
            value = Math.max(this.getMin(),
                Math.min(this.getMax(), value));  // clamp
            this.seekRange.value = this.seekNumber.value = value.toString();
            if (this.callback !== undefined) {
                this.callback(value);
            }
        }
        public getValue(): number {
            return parseInt(this.seekRange.value);
        }

        public getDelay(): number {
            const fps = parseInt(this.fpsInput.value);
            return Math.floor(1000 / fps);
        }
        private resetInterval() {
            if (this.intervalId !== undefined) {
                clearInterval(this.intervalId);
                this.intervalId = undefined;
            }
        }
        public play() {
            this.playButton.removeEventListener('click', this.playClosure);
            this.playButton.addEventListener('click', this.stopClosure);
            this.runIcon.classList.remove('fa-play');
            this.runIcon.classList.add('fa-stop');
            if (this.getValue() == this.getMax()) {  // if last, go to first
                this.setValue(this.getMin());
            }
            this.resetInterval();
            this.intervalId = setInterval(() => {
                if (this.getValue() == this.getMax()) {
                    this.stop();
                } else {
                    this.setValue(this.getValue() + 1);
                }
            }, this.getDelay());
        }
        public stop() {
            this.playButton.removeEventListener('click', this.stopClosure);
            this.playButton.addEventListener('click', this.playClosure);
            this.runIcon.classList.remove('fa-stop');
            this.runIcon.classList.add('fa-play');
            this.resetInterval();
        }
    }

    const loadFile = (io: HTMLInputElement, callback: (value: string) => void) => {
        callback(io.value);
    };

    const saveUrlAsLocalFile = (url: string, filename: string) => {
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;
        const evt = document.createEvent('MouseEvent');
        evt.initEvent("click", true, true);
        anchor.dispatchEvent(evt);
    };

    export class FileExporter {
        constructor(canvas: HTMLCanvasElement, seek: RichSeekBar) {
            const saveAsImage = <HTMLInputElement>document.getElementById("saveAsImage");

            saveAsImage.addEventListener('click', () => {
                saveUrlAsLocalFile(canvas.toDataURL('image/png'), 'canvas.png');
            });
        }
    }
}

module visualizer {
    class InputData {
        public board: string[];

        constructor(content: string) {
            console.log('loading input...');
            const parser = new framework.FileParser('<input-file>', content);
            this.board = [];
            // parse
            while (!parser.isEOF()) {
                let row: string = "";
                while (!parser.isNewline()) {
                    row += parser.getWord();
                }
                this.board.push(row);
                parser.getNewline();
            }
            console.log('complete');
        }
    };

    class OutputData {
        public board: string[];

        constructor(content: string) {
            console.log('loading output...');
            const parser = new framework.FileParser('<output-file>', content);
            // parse
            this.board = [];
            while (!parser.isEOF()) {
                let row: string = "";
                while (!parser.isNewline()) {
                    row += parser.getWord();
                }
                this.board.push(row);
                parser.getNewline();
            }
            console.log('complete');
        }
    };

    class Blob {
        public ch: string;
        public points: [number, number][];
        constructor(ch: string, points: [number, number][]) {
            this.ch = ch;
            this.points = points;
        }
    }

    class Queue<T> {
        _store: T[] = [];
        push(val: T) {
            this._store.push(val);
        }
        pop(): T | undefined {
            return this._store.shift();
        }
    }

    class Result {
        public S: string[];
        public T: string[];
        public dropped: string[];
        public blobs: Blob[];
        public largest_blob: {[key: string]: Blob};
        public score: number;

        private rot_cw(src: string[]): string[] {
            let N: number = src.length;
            let dst: string[] = [];
            for (let i = 0; i < N; i++) {
                let row: string = "";
                for (let j = 0; j < N; j++) {
                    row += src[N - j - 1][i];
                }
                dst.push(row);
            }
            return dst;
        }

        private drop(row_: string): string {
            let row = row_.split('');
            let N: number = row.length;
            let l: number = 0, r: number;
            while (l < N && row[l] != '.') l++;
            for (r = l + 1; r < N; r++) {
                if (row[r] == '-') {
                    l = r + 1;
                    while (l < N && row[l] != '.') l++;
                    r = l;
                }
                else if (row[r] != '.') {
                    row[l++] = row[r];
                    row[r] = '.';
                }
            }
            return row.join('');
        }

        private enum_blobs(S: string[]): Blob[] {
            const di: number[] = [0, -1, 0, 1];
            const dj: number[] = [1, 0, -1, 0];
            let blobs: Blob[] = [];
            let N = S.length;
            let used: Boolean[][] = [];
            for (let i = 0; i < N; i++) {
                used.push([]);
                for (let j = 0; j < N; j++) {
                    used[i].push(false);
                }
            }
            for (let r = 0; r < N; r++) {
                for (let c = 0; c < N; c++) {
                    if (used[r][c] || (S[r][c] != 'o' && S[r][c] != 'x')) continue;
                    let ch: string = S[r][c];
                    let points: [number, number][] = [];
                    let qu: Queue<[number, number]> = new Queue<[number, number]>();
                    used[r][c] = true;
                    qu.push([r, c]);
                    points.push([r, c]);
                    while(qu._store.length > 0) {
                        let [cr, cc] = qu.pop();
                        for (let d = 0; d < 4; d++) {
                            let nr = cr + di[d], nc = cc + dj[d];
                            if (nr < 0 || nr >= N || nc < 0 || nc >= N || used[nr][nc] || S[nr][nc] != ch) continue;
                            used[nr][nc] = true;
                            qu.push([nr, nc]);
                            points.push([nr, nc]);
                        }
                    }
                    let blob: Blob = new Blob(ch, points);
                    blobs.push(blob);
                }
            }
            return blobs;
        }

        private judge(): void {
            let S: string[] = this.S;
            let T: string[] = this.T;
            let N: number = S.length;

            if (T.length != N) {
                alert(`The output must have ${N} lines, but ${T.length}.`);
                return;
            }

            for (let i = 0; i < N; i++) {
                let t: string = T[i];
                if (t.length == N) continue;
                alert(`Invalid row length at line ${i}: ${N} expected, but ${t.length}.`);
                return;
            }

            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    if ((S[i][j] == 'o' || S[i][j] == 'x') && S[i][j] != T[i][j]) {
                        alert(`The value of cell [${i}, ${j}] must be the same as the input.`);
                        return;
                    }
                    if (S[i][j] == '.' && (T[i][j] != '.' && T[i][j] != '+' && T[i][j] != '-')) {
                        alert(`The value of cell [${i}, ${j}] must be one of '.', '+', '-'.`);
                        return;
                    }
                }
            }

            T = this.rot_cw(T);
            for (let i = 0; i < N; i++) T[i] = this.drop(T[i]);
            for (let i = 0; i < 3; i++) T = this.rot_cw(T);

            this.dropped = T;

            this.blobs = this.enum_blobs(T);

            let score: Object = {'o': 0, 'x': 0};
            this.largest_blob = {};
            for(let blob of this.blobs) {
                if(score[blob.ch] < blob.points.length) {
                    score[blob.ch] = blob.points.length;
                    this.largest_blob[blob.ch] = blob;
                }
            }

            this.score = score['o'] + score['x'];
        }

        constructor(input: string[], output: string[]) {
            this.S = input;
            this.T = output;
            this.judge();
        }
    }

    class TesterFrame {
        public board: string[];
        public color: string[][];
        public score: number;
    };

    class Tester {
        public frames: TesterFrame[];

        private create_input_frame(result: Result): TesterFrame {
            let frame: TesterFrame = new TesterFrame();
            frame.board = [];
            frame.color = [];
            for(let src of result.S) {
                let dst = '';
                let cdst = [];
                for (let i = 0; i < src.length; i++) {
                    if(src[i] == 'o') {
                        dst += src[i];
                        cdst.push('rgb(200, 200, 0)');
                    }
                    else if (src[i] == 'x') {
                        dst += src[i];
                        cdst.push('rgb(0, 200, 200)');
                    }
                    else if (src[i] == '+' || src[i] == '-') {
                        dst += src[i];
                        cdst.push('lightgray');
                    }
                    else {
                        dst += ' ';
                        cdst.push('white');
                    }
                }
                frame.board.push(dst);
                frame.color.push(cdst);
            }
            frame.score = result.score;
            return frame;
        }

        private create_output_frame(result: Result): TesterFrame {
            let frame: TesterFrame = new TesterFrame();
            frame.board = [];
            frame.color = [];
            for(let src of result.T) {
                let dst = '';
                let cdst = [];
                for (let i = 0; i < src.length; i++) {
                    if(src[i] == 'o') {
                        dst += src[i];
                        cdst.push('rgb(200, 200, 0)');
                    }
                    else if (src[i] == 'x') {
                        dst += src[i];
                        cdst.push('rgb(0, 200, 200)');
                    }
                    else if (src[i] == '+' || src[i] == '-') {
                        dst += src[i];
                        cdst.push('lightgray');
                    }
                    else {
                        dst += ' ';
                        cdst.push('white');
                    }
                }
                frame.board.push(dst);
                frame.color.push(cdst);
            }
            frame.score = result.score;
            return frame;
        }

        private create_dropped_frame(result: Result): TesterFrame {
            let frame: TesterFrame = new TesterFrame();
            frame.board = [];
            frame.color = [];
            for(let src of result.dropped) {
                let dst = '';
                let cdst = [];
                for (let i = 0; i < src.length; i++) {
                    if(src[i] == 'o') {
                        dst += src[i];
                        cdst.push('rgb(200, 200, 0)');
                    }
                    else if (src[i] == 'x') {
                        dst += src[i];
                        cdst.push('rgb(0, 200, 200)');
                    }
                    else if (src[i] == '+' || src[i] == '-') {
                        dst += src[i];
                        cdst.push('lightgray');
                    }
                    else {
                        dst += ' ';
                        cdst.push('white');
                    }
                }
                frame.board.push(dst);
                frame.color.push(cdst);
            }
            for (let [i, j] of result.largest_blob['o'].points) {
                frame.color[i][j] = 'rgb(255, 255, 0)';
            }
            for (let [i, j] of result.largest_blob['x'].points) {
                frame.color[i][j] = 'rgb(0, 255, 255)'
            }
            frame.score = result.score;
            return frame;
        }

        constructor(inputContent: string, outputContent: string) {
            const input = new InputData(inputContent);
            const output = new OutputData(outputContent);
            const result = new Result(input.board, output.board);

            this.frames = [];
            this.frames.push(this.create_input_frame(result));
            this.frames.push(this.create_output_frame(result));
            this.frames.push(this.create_dropped_frame(result));
        }
    };

    class Visualizer {
        private canvas: HTMLCanvasElement;
        private ctx: CanvasRenderingContext2D;
        private scoreInput: HTMLInputElement;

        constructor() {
            this.canvas = <HTMLCanvasElement>document.getElementById("canvas");  // TODO: IDs should be given as arguments
            const size = 750;
            this.canvas.height = size;  // pixels
            this.canvas.width = size;  // pixels
            this.ctx = this.canvas.getContext('2d');
            if (this.ctx == null) {
                alert('unsupported browser');
            }
            this.scoreInput = <HTMLInputElement>document.getElementById("scoreInput");
        }

        public draw(frame: TesterFrame) {
            this.scoreInput.value = frame.score.toString();

            // update the canvas
            const N = frame.board.length;
            const height = this.canvas.height;
            const width = this.canvas.width;
            const cell_size = height / frame.board.length;

            this.ctx.font = "normal 12px Consolas";

            const drawCell = (i: number, j: number, color: string) => {
                this.ctx.fillStyle = color;
                this.ctx.fillRect(j * cell_size, i * cell_size, cell_size, cell_size);
                this.ctx.strokeStyle = 'rgb(0, 0, 0)';
                this.ctx.lineWidth = 1;
                this.ctx.strokeText(frame.board[i][j], j * cell_size + cell_size / 4, i * cell_size + cell_size * 2 / 3.0);
            };

            this.ctx.fillStyle = "rgb(255, 255, 255)";
            this.ctx.fillRect(0, 0, width, height);

            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    drawCell(i, j, frame.color[i][j]);
                }
            }
        }

        public getCanvas(): HTMLCanvasElement {
            return this.canvas;
        }
    };

    export class App {
        public visualizer: Visualizer;
        public tester: Tester;
        public loader: framework.FileSelector;
        public seek: framework.RichSeekBar;
        public exporter: framework.FileExporter;

        constructor() {
            this.visualizer = new Visualizer();
            this.loader = new framework.FileSelector();
            this.seek = new framework.RichSeekBar();
            this.exporter = new framework.FileExporter(this.visualizer.getCanvas(), this.seek);

            this.seek.callback = (value: number) => {
                if (this.tester !== undefined) {
                    this.visualizer.draw(this.tester.frames[value]);
                }
            };

            this.loader.callback = (inputContent: string, outputContent: string) => {
                this.tester = new Tester(inputContent, outputContent);
                this.seek.setMinMax(0, this.tester.frames.length - 1);
                this.seek.setValue(0);
                this.seek.play();
            };
        }

        public run() {
            this.loader.clickReloadButton();
        }
    }
}

window.onload = () => {
    let app = new visualizer.App();
    app.run();
};
