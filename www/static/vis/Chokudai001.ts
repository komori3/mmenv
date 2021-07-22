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
                if(line.length == 0) continue;
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
        private inputFile: HTMLInputElement;
        private outputFile: HTMLInputElement;
        private reloadButton: HTMLInputElement;

        constructor() {
            this.inputText = <HTMLInputElement> document.getElementById("inputText")
            this.outputText = <HTMLInputElement> document.getElementById("outputText");
            this.reloadButton = <HTMLInputElement> document.getElementById("reloadButton");
            this.reloadFilesClosure = () => { this.reloadFiles(); };
            this. inputText.addEventListener('change', this.reloadFilesClosure);
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
                    this. inputText.removeEventListener('change', this.reloadFilesClosure);
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
            this.seekRange  = <HTMLInputElement> document.getElementById("seekRange");
            this.seekNumber = <HTMLInputElement> document.getElementById("seekNumber");
            this.fpsInput = <HTMLInputElement> document.getElementById("fpsInput");
            this.firstButton = <HTMLInputElement> document.getElementById("firstButton");
            this.prevButton = <HTMLInputElement> document.getElementById("prevButton");
            this.playButton = <HTMLInputElement> document.getElementById("playButton");
            this.nextButton = <HTMLInputElement> document.getElementById("nextButton");
            this.lastButton = <HTMLInputElement> document.getElementById("lastButton");
            this.runIcon = document.getElementById("runIcon");
            this.intervalId = null;

            this.setMinMax(-1, -1);
            this.seekRange .addEventListener('change', () => { this.setValue(parseInt(this.seekRange .value)); });
            this.seekNumber.addEventListener('change', () => { this.setValue(parseInt(this.seekNumber.value)); });
            this.seekRange .addEventListener('input',  () => { this.setValue(parseInt(this.seekRange .value)); });
            this.seekNumber.addEventListener('input',  () => { this.setValue(parseInt(this.seekNumber.value)); });
            this.fpsInput.addEventListener('change', () => { if (this.intervalId !== null) { this.play(); } });
            this.firstButton.addEventListener('click', () => { this.stop(); this.setValue(this.getMin()); });
            this.prevButton .addEventListener('click', () => { this.stop(); this.setValue(this.getValue() - 1); });
            this.nextButton .addEventListener('click', () => { this.stop(); this.setValue(this.getValue() + 1); });
            this.lastButton .addEventListener('click', () => { this.stop(); this.setValue(this.getMax()); });
            this.playClosure = () => { this.play(); };
            this.stopClosure = () => { this.stop(); };
            this.playButton.addEventListener('click', this.playClosure);
        }

        public setMinMax(min: number, max: number) {
            this.seekRange.min   = this.seekNumber.min   = min.toString();
            this.seekRange.max   = this.seekNumber.max   = max.toString();
            this.seekRange.step  = this.seekNumber.step  = '1';
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
            this.playButton.   addEventListener('click', this.stopClosure);
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
            this.playButton.   addEventListener('click', this.playClosure);
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
            const saveAsImage = <HTMLInputElement> document.getElementById("saveAsImage");
            const saveAsVideo = <HTMLInputElement> document.getElementById("saveAsVideo");

            saveAsImage.addEventListener('click', () => {
                saveUrlAsLocalFile(canvas.toDataURL('image/png'), 'canvas.png');
            });

            saveAsVideo.addEventListener('click', () => {
                if (location.href.match(new RegExp('^file://'))) {
                    alert('to use this feature, you must re-open this file via "http://", instead of "file://". e.g. you can use "$ python -m SimpleHTTPServer 8000"');
                }
                seek.stop();
                const gif = new GIF();
                for (let i = seek.getMin(); i < seek.getMax(); ++ i) {
                    seek.setValue(i);
                    gif.addFrame(canvas, { copy: true, delay: seek.getDelay() });
                }
                gif.on('finished', function(blob) {
                    saveUrlAsLocalFile(URL.createObjectURL(blob), 'canvas.gif');
                });
                gif.render();
                alert('please wait for a while, about 2 minutes.');
            });
        }
    }
}

module visualizer {
    class InputData {
        public board: number[][];

        constructor(content: string) {
            console.log('loading input...');
            const parser = new framework.FileParser('<input-file>', content);
            this.board = [];
            // parse
            while(!parser.isEOF()) {
                let row: number[] = [];
                while(!parser.isNewline()) {
                    row.push(parser.getInt());
                }
                console.log(row.length);
                this.board.push(row);
                parser.getNewline();
            }
            console.log('complete');
        }
    };

    class OutputData {
        public commands: [number, number][];

        constructor(content: string) {
            console.log('loading output...');
            const parser = new framework.FileParser('<output-file>', content);
            // parse
            this.commands = [];
            while(!parser.isEOF()) {
                let y: number = parser.getInt();
                let x: number = parser.getInt();
                this.commands.push([y, x]);
                parser.getNewline();
            }
            console.log(this.commands.length);
            console.log('complete');
        }
    };

    class TesterFrame {
        public board: number[][];
        public previousFrame: TesterFrame | null;
        public age: number;
        public command: [number, number][];
        public score: number;

        constructor(input: InputData, score: number);
        constructor(frame: TesterFrame, command: [number, number][]);
        constructor(something1: any, something2?: any) {
            if (something1 instanceof InputData) {  // initial frame
                const input = something1 as InputData;
                const score = something2 as number;
                this.board = input.board;
                this.previousFrame = null;
                this.age = 0;
                this.command = [];
                this.score = score;
            } else if (something1 instanceof TesterFrame) {  // successor frame
                this.previousFrame = something1 as TesterFrame;
                this.command = something2 as [number, number][];
                this.board = JSON.parse(JSON.stringify(this.previousFrame.board)); // deep copy
                this.age = this.previousFrame.age + 1;
                this.score = this.previousFrame.score;
                // apply the command
                for (let [i, j] of this.command) {
                    this.board[i][j]--;
                }
            }
        }
    };

    class CommandGenerator {
        private N: number;
        public commands: [number, number][][];

        private isInside(i: number, j: number): boolean {
            return 0 <= i && i < this.N && 0 <= j && j < this.N;
        }
        private isAdjacent(i1: number, j1: number, i2: number, j2: number): boolean {
            return Math.abs(i1 - i2) + Math.abs(j1 - j2) === 1;
        }

        constructor(input: InputData, output: OutputData) {
            let board: number[][] = JSON.parse(JSON.stringify(input.board));
            const N: number = board.length;
            this.N = N;
            for(let i: number = 0; i < N; i++) {
                let row: number[] = board[i];
                if (row.length != N) {
                    alert(`<CommandGenerator>: row[${i}].length is expected to be ${N}, but ${row.length}`);
                    return;
                }
            }
            this.commands = [];
            let moves: [number, number][] = output.commands;
            console.log(moves.length);
            let pi: number = -2, pj: number = -2;
            for (let row: number = 0; row < moves.length; row++) {
                let i: number = moves[row][0] - 1, j: number = moves[row][1] - 1;
                if (!this.isInside(i, j)) {
                    alert(`Invalid cell [${i+1}, ${j+1}] is specified at row ${row}.`);
                    return;
                }
                if (board[i][j] == 0) {
                    alert(`An operation on cell [${i+1}, ${j+1}] with a value 0 was found in row ${row}`);
                    return;
                }
                if (!this.isAdjacent(pi, pj, i, j) || board[pi][pj] != board[i][j]) {
                    this.commands.push([]);
                }
                board[i][j]--;
                this.commands[this.commands.length - 1].push([i, j]); // to 0-indexed
                pi = i; pj = j;
            }
            console.log(this.commands.length);
        }
    }

    class Tester {
        public frames: TesterFrame[];
        constructor(inputContent: string, outputContent: string) {
            const input  = new  InputData( inputContent);
            const output = new OutputData(outputContent);

            const cmdgen = new CommandGenerator(input, output);
            const commands = cmdgen.commands;

            this.frames = [ new TesterFrame(input, 100000 - commands.length) ];
            for (const command of commands) {
                let lastFrame = this.frames[this.frames.length - 1];
                this.frames.push( new TesterFrame(lastFrame, command) );
            }
        }
    };

    class Visualizer {
        private canvas: HTMLCanvasElement;
        private ctx: CanvasRenderingContext2D;
        private scoreInput: HTMLInputElement;

        constructor() {
            this.canvas = <HTMLCanvasElement> document.getElementById("canvas");  // TODO: IDs should be given as arguments
            const size = 750;
            this.canvas.height = size;  // pixels
            this.canvas.width  = size;  // pixels
            this.ctx = this.canvas.getContext('2d');
            if (this.ctx == null) {
                alert('unsupported browser');
            }
            this.scoreInput = <HTMLInputElement> document.getElementById("scoreInput");
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
                this.ctx.strokeText(frame.board[i][j].toString(), j * cell_size + cell_size / 4, i * cell_size + cell_size * 2 / 3.0);
            };

            const getPoint = (p: [number, number]) => {
                return [p[1] * cell_size + cell_size / 2, p[0] * cell_size + cell_size / 2];
            }

            const drawCircle = (p: [number, number]) => {
                this.ctx.beginPath();
                let [x, y] = getPoint(p);
                this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
                this.ctx.fill();
            }

            const drawLine = (p1: [number, number], p2: [number, number]) => {
                this.ctx.beginPath();
                let [x1, y1] = getPoint(p1);
                let [x2, y2] = getPoint(p2);
                this.ctx.moveTo(x1, y1);
                this.ctx.lineTo(x2, y2);
                this.ctx.stroke();
            }

            const drawPath = (path: [number, number][]) => {
                if (path.length == 0) return;
                this.ctx.fillStyle = 'black';
                drawCircle(path[0]);
                for(let i: number = 1; i < path.length; i++) {
                    drawLine(path[i - 1], path[i]);
                    drawCircle(path[i]);
                }
            }

            this.ctx.fillStyle = "rgb(255, 255, 255)";
            this.ctx.fillRect(0, 0, width, height);

            // draw entities
            const getColor = (i: number, j: number) => {
                if (frame.board[i][j] == 0) {
                    return 'rgb(150, 150, 150)';
                }
                let ir: number = Math.round(frame.board[i][j] * 255.0 / 100.0);
                return `rgb(${255}, ${255 - ir}, ${255 - ir})`;
            }

            for (let i = 0; i < N; i++) {
                for(let j = 0; j < N; j++) {
                    drawCell(i, j, getColor(i, j));
                }
            }

            this.ctx.strokeStyle = 'black';
            this.ctx.lineWidth = 5;
            drawPath(frame.command);
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
