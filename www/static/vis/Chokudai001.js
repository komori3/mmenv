var framework;
(function (framework) {
    var FileParser = /** @class */ (function () {
        function FileParser(filename, content) {
            this.filename = filename;
            this.content = [];
            for (var _i = 0, _a = content.split('\n'); _i < _a.length; _i++) {
                var line = _a[_i];
                if (line.length == 0)
                    continue;
                var words = line.trim().split(new RegExp('\\s+'));
                this.content.push(words);
            }
            this.y = 0;
            this.x = 0;
        }
        FileParser.prototype.getWord = function () {
            if (this.content.length <= this.y) {
                this.reportError('a word expected, but EOF');
            }
            if (this.content[this.y].length <= this.x) {
                this.reportError('a word expected, but newline');
            }
            var word = this.content[this.y][this.x];
            this.x += 1;
            return word;
        };
        FileParser.prototype.getInt = function () {
            var word = this.getWord();
            if (!word.match(new RegExp('^[-+]?[0-9]+$'))) {
                this.reportError("a number expected, but word " + JSON.stringify(this.content[this.y][this.x]));
            }
            return parseInt(word);
        };
        FileParser.prototype.getNewline = function () {
            if (this.content.length <= this.y) {
                this.reportError('newline expected, but EOF');
            }
            if (this.x < this.content[this.y].length) {
                this.reportError("newline expected, but word " + JSON.stringify(this.content[this.y][this.x]));
            }
            this.x = 0;
            this.y += 1;
        };
        FileParser.prototype.isNewline = function () {
            return this.content[this.y].length <= this.x;
        };
        FileParser.prototype.isEOF = function () {
            return this.content.length <= this.y;
        };
        FileParser.prototype.unwind = function () {
            if (this.x == 0) {
                this.y -= 1;
                this.x = this.content[this.y].length - 1;
            }
            else {
                this.x -= 1;
            }
        };
        FileParser.prototype.reportError = function (msg) {
            msg = this.filename + ": line " + (this.y + 1) + ": " + msg;
            alert(msg);
            throw new Error(msg);
        };
        return FileParser;
    }());
    framework.FileParser = FileParser;
    var FileSelector = /** @class */ (function () {
        function FileSelector() {
            var _this = this;
            this.inputText = document.getElementById("inputText");
            this.outputText = document.getElementById("outputText");
            this.reloadButton = document.getElementById("reloadButton");
            this.reloadFilesClosure = function () { _this.reloadFiles(); };
            this.inputText.addEventListener('change', this.reloadFilesClosure);
            this.outputText.addEventListener('change', this.reloadFilesClosure);
            this.reloadButton.addEventListener('click', this.reloadFilesClosure);
        }
        FileSelector.prototype.clickReloadButton = function () {
            this.reloadButton.dispatchEvent(new Event('click'));
        };
        FileSelector.prototype.reloadFiles = function () {
            var _this = this;
            if (this.inputText.value.length == 0 || this.outputText.value.length == 0)
                return;
            loadFile(this.inputText, function (inputContent) {
                loadFile(_this.outputText, function (outputContent) {
                    _this.inputText.removeEventListener('change', _this.reloadFilesClosure);
                    _this.outputText.removeEventListener('change', _this.reloadFilesClosure);
                    _this.reloadButton.classList.remove('disabled');
                    if (_this.callback !== undefined) {
                        _this.callback(inputContent, outputContent);
                    }
                });
            });
        };
        return FileSelector;
    }());
    framework.FileSelector = FileSelector;
    var RichSeekBar = /** @class */ (function () {
        function RichSeekBar() {
            var _this = this;
            this.seekRange = document.getElementById("seekRange");
            this.seekNumber = document.getElementById("seekNumber");
            this.fpsInput = document.getElementById("fpsInput");
            this.firstButton = document.getElementById("firstButton");
            this.prevButton = document.getElementById("prevButton");
            this.playButton = document.getElementById("playButton");
            this.nextButton = document.getElementById("nextButton");
            this.lastButton = document.getElementById("lastButton");
            this.runIcon = document.getElementById("runIcon");
            this.intervalId = null;
            this.setMinMax(-1, -1);
            this.seekRange.addEventListener('change', function () { _this.setValue(parseInt(_this.seekRange.value)); });
            this.seekNumber.addEventListener('change', function () { _this.setValue(parseInt(_this.seekNumber.value)); });
            this.seekRange.addEventListener('input', function () { _this.setValue(parseInt(_this.seekRange.value)); });
            this.seekNumber.addEventListener('input', function () { _this.setValue(parseInt(_this.seekNumber.value)); });
            this.fpsInput.addEventListener('change', function () { if (_this.intervalId !== null) {
                _this.play();
            } });
            this.firstButton.addEventListener('click', function () { _this.stop(); _this.setValue(_this.getMin()); });
            this.prevButton.addEventListener('click', function () { _this.stop(); _this.setValue(_this.getValue() - 1); });
            this.nextButton.addEventListener('click', function () { _this.stop(); _this.setValue(_this.getValue() + 1); });
            this.lastButton.addEventListener('click', function () { _this.stop(); _this.setValue(_this.getMax()); });
            this.playClosure = function () { _this.play(); };
            this.stopClosure = function () { _this.stop(); };
            this.playButton.addEventListener('click', this.playClosure);
        }
        RichSeekBar.prototype.setMinMax = function (min, max) {
            this.seekRange.min = this.seekNumber.min = min.toString();
            this.seekRange.max = this.seekNumber.max = max.toString();
            this.seekRange.step = this.seekNumber.step = '1';
            this.setValue(min);
        };
        RichSeekBar.prototype.getMin = function () {
            return parseInt(this.seekRange.min);
        };
        RichSeekBar.prototype.getMax = function () {
            return parseInt(this.seekRange.max);
        };
        RichSeekBar.prototype.setValue = function (value) {
            value = Math.max(this.getMin(), Math.min(this.getMax(), value)); // clamp
            this.seekRange.value = this.seekNumber.value = value.toString();
            if (this.callback !== undefined) {
                this.callback(value);
            }
        };
        RichSeekBar.prototype.getValue = function () {
            return parseInt(this.seekRange.value);
        };
        RichSeekBar.prototype.getDelay = function () {
            var fps = parseInt(this.fpsInput.value);
            return Math.floor(1000 / fps);
        };
        RichSeekBar.prototype.resetInterval = function () {
            if (this.intervalId !== undefined) {
                clearInterval(this.intervalId);
                this.intervalId = undefined;
            }
        };
        RichSeekBar.prototype.play = function () {
            var _this = this;
            this.playButton.removeEventListener('click', this.playClosure);
            this.playButton.addEventListener('click', this.stopClosure);
            this.runIcon.classList.remove('fa-play');
            this.runIcon.classList.add('fa-stop');
            if (this.getValue() == this.getMax()) { // if last, go to first
                this.setValue(this.getMin());
            }
            this.resetInterval();
            this.intervalId = setInterval(function () {
                if (_this.getValue() == _this.getMax()) {
                    _this.stop();
                }
                else {
                    _this.setValue(_this.getValue() + 1);
                }
            }, this.getDelay());
        };
        RichSeekBar.prototype.stop = function () {
            this.playButton.removeEventListener('click', this.stopClosure);
            this.playButton.addEventListener('click', this.playClosure);
            this.runIcon.classList.remove('fa-stop');
            this.runIcon.classList.add('fa-play');
            this.resetInterval();
        };
        return RichSeekBar;
    }());
    framework.RichSeekBar = RichSeekBar;
    var loadFile = function (io, callback) {
        callback(io.value);
    };
    var saveUrlAsLocalFile = function (url, filename) {
        var anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;
        var evt = document.createEvent('MouseEvent');
        evt.initEvent("click", true, true);
        anchor.dispatchEvent(evt);
    };
    var FileExporter = /** @class */ (function () {
        function FileExporter(canvas, seek) {
            var saveAsImage = document.getElementById("saveAsImage");
            var saveAsVideo = document.getElementById("saveAsVideo");
            saveAsImage.addEventListener('click', function () {
                saveUrlAsLocalFile(canvas.toDataURL('image/png'), 'canvas.png');
            });
            saveAsVideo.addEventListener('click', function () {
                if (location.href.match(new RegExp('^file://'))) {
                    alert('to use this feature, you must re-open this file via "http://", instead of "file://". e.g. you can use "$ python -m SimpleHTTPServer 8000"');
                }
                seek.stop();
                var gif = new GIF();
                for (var i = seek.getMin(); i < seek.getMax(); ++i) {
                    seek.setValue(i);
                    gif.addFrame(canvas, { copy: true, delay: seek.getDelay() });
                }
                gif.on('finished', function (blob) {
                    saveUrlAsLocalFile(URL.createObjectURL(blob), 'canvas.gif');
                });
                gif.render();
                alert('please wait for a while, about 2 minutes.');
            });
        }
        return FileExporter;
    }());
    framework.FileExporter = FileExporter;
})(framework || (framework = {}));
var visualizer;
(function (visualizer) {
    var InputData = /** @class */ (function () {
        function InputData(content) {
            console.log('loading input...');
            var parser = new framework.FileParser('<input-file>', content);
            this.board = [];
            // parse
            while (!parser.isEOF()) {
                var row = [];
                while (!parser.isNewline()) {
                    row.push(parser.getInt());
                }
                console.log(row.length);
                this.board.push(row);
                parser.getNewline();
            }
            console.log('complete');
        }
        return InputData;
    }());
    ;
    var OutputData = /** @class */ (function () {
        function OutputData(content) {
            console.log('loading output...');
            var parser = new framework.FileParser('<output-file>', content);
            // parse
            this.commands = [];
            while (!parser.isEOF()) {
                var y = parser.getInt();
                var x = parser.getInt();
                this.commands.push([y, x]);
                parser.getNewline();
            }
            console.log(this.commands.length);
            console.log('complete');
        }
        return OutputData;
    }());
    ;
    var TesterFrame = /** @class */ (function () {
        function TesterFrame(something1, something2) {
            if (something1 instanceof InputData) { // initial frame
                var input = something1;
                var score = something2;
                this.board = input.board;
                this.previousFrame = null;
                this.age = 0;
                this.command = [];
                this.score = score;
            }
            else if (something1 instanceof TesterFrame) { // successor frame
                this.previousFrame = something1;
                this.command = something2;
                this.board = JSON.parse(JSON.stringify(this.previousFrame.board)); // deep copy
                this.age = this.previousFrame.age + 1;
                this.score = this.previousFrame.score;
                // apply the command
                for (var _i = 0, _a = this.command; _i < _a.length; _i++) {
                    var _b = _a[_i], i = _b[0], j = _b[1];
                    this.board[i][j]--;
                }
            }
        }
        return TesterFrame;
    }());
    ;
    var CommandGenerator = /** @class */ (function () {
        function CommandGenerator(input, output) {
            var board = JSON.parse(JSON.stringify(input.board));
            var N = board.length;
            this.N = N;
            for (var i = 0; i < N; i++) {
                var row = board[i];
                if (row.length != N) {
                    alert("<CommandGenerator>: row[" + i + "].length is expected to be " + N + ", but " + row.length);
                    return;
                }
            }
            this.commands = [];
            var moves = output.commands;
            console.log(moves.length);
            var pi = -2, pj = -2;
            for (var row = 0; row < moves.length; row++) {
                var i = moves[row][0] - 1, j = moves[row][1] - 1;
                if (!this.isInside(i, j)) {
                    alert("Invalid cell [" + (i + 1) + ", " + (j + 1) + "] is specified at row " + row + ".");
                    return;
                }
                if (board[i][j] == 0) {
                    alert("An operation on cell [" + (i + 1) + ", " + (j + 1) + "] with a value 0 was found in row " + row);
                    return;
                }
                if (!this.isAdjacent(pi, pj, i, j) || board[pi][pj] != board[i][j]) {
                    this.commands.push([]);
                }
                board[i][j]--;
                this.commands[this.commands.length - 1].push([i, j]); // to 0-indexed
                pi = i;
                pj = j;
            }
            console.log(this.commands.length);
        }
        CommandGenerator.prototype.isInside = function (i, j) {
            return 0 <= i && i < this.N && 0 <= j && j < this.N;
        };
        CommandGenerator.prototype.isAdjacent = function (i1, j1, i2, j2) {
            return Math.abs(i1 - i2) + Math.abs(j1 - j2) === 1;
        };
        return CommandGenerator;
    }());
    var Tester = /** @class */ (function () {
        function Tester(inputContent, outputContent) {
            var input = new InputData(inputContent);
            var output = new OutputData(outputContent);
            var cmdgen = new CommandGenerator(input, output);
            var commands = cmdgen.commands;
            this.frames = [new TesterFrame(input, 100000 - commands.length)];
            for (var _i = 0, commands_1 = commands; _i < commands_1.length; _i++) {
                var command = commands_1[_i];
                var lastFrame = this.frames[this.frames.length - 1];
                this.frames.push(new TesterFrame(lastFrame, command));
            }
        }
        return Tester;
    }());
    ;
    var Visualizer = /** @class */ (function () {
        function Visualizer() {
            this.canvas = document.getElementById("canvas"); // TODO: IDs should be given as arguments
            var size = 750;
            this.canvas.height = size; // pixels
            this.canvas.width = size; // pixels
            this.ctx = this.canvas.getContext('2d');
            if (this.ctx == null) {
                alert('unsupported browser');
            }
            this.scoreInput = document.getElementById("scoreInput");
        }
        Visualizer.prototype.draw = function (frame) {
            var _this = this;
            this.scoreInput.value = frame.score.toString();
            // update the canvas
            var N = frame.board.length;
            var height = this.canvas.height;
            var width = this.canvas.width;
            var cell_size = height / frame.board.length;
            this.ctx.font = "normal 12px Consolas";
            var drawCell = function (i, j, color) {
                _this.ctx.fillStyle = color;
                _this.ctx.fillRect(j * cell_size, i * cell_size, cell_size, cell_size);
                _this.ctx.strokeStyle = 'rgb(0, 0, 0)';
                _this.ctx.lineWidth = 1;
                _this.ctx.strokeText(frame.board[i][j].toString(), j * cell_size + cell_size / 4, i * cell_size + cell_size * 2 / 3.0);
            };
            var getPoint = function (p) {
                return [p[1] * cell_size + cell_size / 2, p[0] * cell_size + cell_size / 2];
            };
            var drawCircle = function (p) {
                _this.ctx.beginPath();
                var _a = getPoint(p), x = _a[0], y = _a[1];
                _this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
                _this.ctx.fill();
            };
            var drawLine = function (p1, p2) {
                _this.ctx.beginPath();
                var _a = getPoint(p1), x1 = _a[0], y1 = _a[1];
                var _b = getPoint(p2), x2 = _b[0], y2 = _b[1];
                _this.ctx.moveTo(x1, y1);
                _this.ctx.lineTo(x2, y2);
                _this.ctx.stroke();
            };
            var drawPath = function (path) {
                if (path.length == 0)
                    return;
                _this.ctx.fillStyle = 'black';
                drawCircle(path[0]);
                for (var i = 1; i < path.length; i++) {
                    drawLine(path[i - 1], path[i]);
                    drawCircle(path[i]);
                }
            };
            this.ctx.fillStyle = "rgb(255, 255, 255)";
            this.ctx.fillRect(0, 0, width, height);
            // draw entities
            var getColor = function (i, j) {
                if (frame.board[i][j] == 0) {
                    return 'rgb(150, 150, 150)';
                }
                var ir = Math.round(frame.board[i][j] * 255.0 / 100.0);
                return "rgb(" + 255 + ", " + (255 - ir) + ", " + (255 - ir) + ")";
            };
            for (var i = 0; i < N; i++) {
                for (var j = 0; j < N; j++) {
                    drawCell(i, j, getColor(i, j));
                }
            }
            this.ctx.strokeStyle = 'black';
            this.ctx.lineWidth = 5;
            drawPath(frame.command);
        };
        Visualizer.prototype.getCanvas = function () {
            return this.canvas;
        };
        return Visualizer;
    }());
    ;
    var App = /** @class */ (function () {
        function App() {
            var _this = this;
            this.visualizer = new Visualizer();
            this.loader = new framework.FileSelector();
            this.seek = new framework.RichSeekBar();
            this.exporter = new framework.FileExporter(this.visualizer.getCanvas(), this.seek);
            this.seek.callback = function (value) {
                if (_this.tester !== undefined) {
                    _this.visualizer.draw(_this.tester.frames[value]);
                }
            };
            this.loader.callback = function (inputContent, outputContent) {
                _this.tester = new Tester(inputContent, outputContent);
                _this.seek.setMinMax(0, _this.tester.frames.length - 1);
                _this.seek.setValue(0);
                _this.seek.play();
            };
        }
        App.prototype.run = function () {
            this.loader.clickReloadButton();
        };
        return App;
    }());
    visualizer.App = App;
})(visualizer || (visualizer = {}));
window.onload = function () {
    var app = new visualizer.App();
    app.run();
};
