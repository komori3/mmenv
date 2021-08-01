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
            saveAsImage.addEventListener('click', function () {
                saveUrlAsLocalFile(canvas.toDataURL('image/png'), 'canvas.png');
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
                var row = "";
                while (!parser.isNewline()) {
                    row += parser.getWord();
                }
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
            this.board = [];
            while (!parser.isEOF()) {
                var row = "";
                while (!parser.isNewline()) {
                    row += parser.getWord();
                }
                this.board.push(row);
                parser.getNewline();
            }
            console.log('complete');
        }
        return OutputData;
    }());
    ;
    var Blob = /** @class */ (function () {
        function Blob(ch, points) {
            this.ch = ch;
            this.points = points;
        }
        return Blob;
    }());
    var Queue = /** @class */ (function () {
        function Queue() {
            this._store = [];
        }
        Queue.prototype.push = function (val) {
            this._store.push(val);
        };
        Queue.prototype.pop = function () {
            return this._store.shift();
        };
        return Queue;
    }());
    var Result = /** @class */ (function () {
        function Result(input, output) {
            this.S = input;
            this.T = output;
            this.judge();
        }
        Result.prototype.rot_cw = function (src) {
            var N = src.length;
            var dst = [];
            for (var i = 0; i < N; i++) {
                var row = "";
                for (var j = 0; j < N; j++) {
                    row += src[N - j - 1][i];
                }
                dst.push(row);
            }
            return dst;
        };
        Result.prototype.drop = function (row_) {
            var row = row_.split('');
            var N = row.length;
            var l = 0, r;
            while (l < N && row[l] != '.')
                l++;
            for (r = l + 1; r < N; r++) {
                if (row[r] == '-') {
                    l = r + 1;
                    while (l < N && row[l] != '.')
                        l++;
                    r = l;
                }
                else if (row[r] != '.') {
                    row[l++] = row[r];
                    row[r] = '.';
                }
            }
            return row.join('');
        };
        Result.prototype.enum_blobs = function (S) {
            var di = [0, -1, 0, 1];
            var dj = [1, 0, -1, 0];
            var blobs = [];
            var N = S.length;
            var used = [];
            for (var i = 0; i < N; i++) {
                used.push([]);
                for (var j = 0; j < N; j++) {
                    used[i].push(false);
                }
            }
            for (var r = 0; r < N; r++) {
                for (var c = 0; c < N; c++) {
                    if (used[r][c] || (S[r][c] != 'o' && S[r][c] != 'x'))
                        continue;
                    var ch = S[r][c];
                    var points = [];
                    var qu = new Queue();
                    used[r][c] = true;
                    qu.push([r, c]);
                    points.push([r, c]);
                    while (qu._store.length > 0) {
                        var _a = qu.pop(), cr = _a[0], cc = _a[1];
                        for (var d = 0; d < 4; d++) {
                            var nr = cr + di[d], nc = cc + dj[d];
                            if (nr < 0 || nr >= N || nc < 0 || nc >= N || used[nr][nc] || S[nr][nc] != ch)
                                continue;
                            used[nr][nc] = true;
                            qu.push([nr, nc]);
                            points.push([nr, nc]);
                        }
                    }
                    var blob = new Blob(ch, points);
                    blobs.push(blob);
                }
            }
            return blobs;
        };
        Result.prototype.judge = function () {
            var S = this.S;
            var T = this.T;
            var N = S.length;
            if (T.length != N) {
                alert("The output must have " + N + " lines, but " + T.length + ".");
                return;
            }
            for (var i = 0; i < N; i++) {
                var t = T[i];
                if (t.length == N)
                    continue;
                alert("Invalid row length at line " + i + ": " + N + " expected, but " + t.length + ".");
                return;
            }
            for (var i = 0; i < N; i++) {
                for (var j = 0; j < N; j++) {
                    if ((S[i][j] == 'o' || S[i][j] == 'x') && S[i][j] != T[i][j]) {
                        alert("The value of cell [" + i + ", " + j + "] must be the same as the input.");
                        return;
                    }
                    if (S[i][j] == '.' && (T[i][j] != '.' && T[i][j] != '+' && T[i][j] != '-')) {
                        alert("The value of cell [" + i + ", " + j + "] must be one of '.', '+', '-'.");
                        return;
                    }
                }
            }
            T = this.rot_cw(T);
            for (var i = 0; i < N; i++)
                T[i] = this.drop(T[i]);
            for (var i = 0; i < 3; i++)
                T = this.rot_cw(T);
            this.dropped = T;
            this.blobs = this.enum_blobs(T);
            var score = { 'o': 0, 'x': 0 };
            this.largest_blob = {};
            for (var _i = 0, _a = this.blobs; _i < _a.length; _i++) {
                var blob = _a[_i];
                if (score[blob.ch] < blob.points.length) {
                    score[blob.ch] = blob.points.length;
                    this.largest_blob[blob.ch] = blob;
                }
            }
            this.score = score['o'] + score['x'];
        };
        return Result;
    }());
    var TesterFrame = /** @class */ (function () {
        function TesterFrame() {
        }
        return TesterFrame;
    }());
    ;
    var Tester = /** @class */ (function () {
        function Tester(inputContent, outputContent) {
            var input = new InputData(inputContent);
            var output = new OutputData(outputContent);
            var result = new Result(input.board, output.board);
            this.frames = [];
            this.frames.push(this.create_input_frame(result));
            this.frames.push(this.create_output_frame(result));
            this.frames.push(this.create_dropped_frame(result));
        }
        Tester.prototype.create_input_frame = function (result) {
            var frame = new TesterFrame();
            frame.board = [];
            frame.color = [];
            for (var _i = 0, _a = result.S; _i < _a.length; _i++) {
                var src = _a[_i];
                var dst = '';
                var cdst = [];
                for (var i = 0; i < src.length; i++) {
                    if (src[i] == 'o') {
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
        };
        Tester.prototype.create_output_frame = function (result) {
            var frame = new TesterFrame();
            frame.board = [];
            frame.color = [];
            for (var _i = 0, _a = result.T; _i < _a.length; _i++) {
                var src = _a[_i];
                var dst = '';
                var cdst = [];
                for (var i = 0; i < src.length; i++) {
                    if (src[i] == 'o') {
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
        };
        Tester.prototype.create_dropped_frame = function (result) {
            var frame = new TesterFrame();
            frame.board = [];
            frame.color = [];
            for (var _i = 0, _a = result.dropped; _i < _a.length; _i++) {
                var src = _a[_i];
                var dst = '';
                var cdst = [];
                for (var i = 0; i < src.length; i++) {
                    if (src[i] == 'o') {
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
            for (var _b = 0, _c = result.largest_blob['o'].points; _b < _c.length; _b++) {
                var _d = _c[_b], i = _d[0], j = _d[1];
                frame.color[i][j] = 'rgb(255, 255, 0)';
            }
            for (var _e = 0, _f = result.largest_blob['x'].points; _e < _f.length; _e++) {
                var _g = _f[_e], i = _g[0], j = _g[1];
                frame.color[i][j] = 'rgb(0, 255, 255)';
            }
            frame.score = result.score;
            return frame;
        };
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
                _this.ctx.strokeText(frame.board[i][j], j * cell_size + cell_size / 4, i * cell_size + cell_size * 2 / 3.0);
            };
            this.ctx.fillStyle = "rgb(255, 255, 255)";
            this.ctx.fillRect(0, 0, width, height);
            for (var i = 0; i < N; i++) {
                for (var j = 0; j < N; j++) {
                    drawCell(i, j, frame.color[i][j]);
                }
            }
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
