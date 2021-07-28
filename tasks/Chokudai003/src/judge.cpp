#define _CRT_SECURE_NO_WARNINGS
#include "CLI11.hpp"
#include "atcoder-util.hpp"
#include "contest_types.hpp"
#include "json.hpp"
#include <filesystem>

const std::string CC_RESET = "\x1b[0m";
const std::string CC_BLACK = "\x1b[40m";
const std::string CC_RED = "\x1b[41m";
const std::string CC_GREEN = "\x1b[42m";
const std::string CC_YELLOW = "\x1b[43m";
const std::string CC_BLUE = "\x1b[44m";
const std::string CC_MAGENTA = "\x1b[45m";
const std::string CC_CYAN = "\x1b[46m";
const std::string CC_WHITE = "\x1b[47m";

struct SJudgeResult;
using SJudgeResultPtr = std::shared_ptr<SJudgeResult>;
struct SJudgeResult {
    int N;
    SProblemPtr prob;
    SSolutionPtr sol;
    int score;
    bool is_valid;
};

using Point = std::pair<int, int>;

struct Blob {
    char ch;
    std::vector<Point> points;
    std::string str() const {
        std::ostringstream oss;
        oss << "Blob [ch=" << ch << ", size=" << points.size() << ", points=" << points << "]";
        return oss.str();
    }
    friend std::ostream& operator<<(std::ostream& o, const Blob& b) {
        o << b.str();
        return o;
    }
};

std::vector<Blob> enum_blobs(const std::vector<std::string>& S) {
    static constexpr int di[] = {0, -1, 0, 1};
    static constexpr int dj[] = {1, 0, -1, 0};
    std::vector<Blob> blobs;
    int N = S.size();
    std::vector<std::vector<bool>> used(N, std::vector<bool>(N, false));
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (S[r][c] != 'o' && S[r][c] != 'x') continue;
            Blob blob; blob.ch = S[r][c];
            std::queue<int> qu;
            used[r][c] = true;
            qu.push(r << 6 | c);
            blob.points.emplace_back(r, c);
            while (!qu.empty()) {
                int crcc = qu.front(); qu.pop();
                int cr = crcc >> 6, cc = crcc & 0b111111;
                for (int d = 0; d < 4; d++) {
                    int nr = cr + di[d], nc = cc + dj[d];
                    if (nr < 0 || nr >= N || nc < 0 || nc >= N || used[nr][nc] || S[nr][nc] != blob.ch) continue;
                    used[nr][nc] = true;
                    qu.push(nr << 6 | nc);
                    blob.points.emplace_back(nr, nc);
                }
            }
            blobs.push_back(blob);
        }
    }
    return blobs;
}

std::pair<int, int> largest_blob(const std::vector<std::string>& S, char ch) {
    static constexpr int di[] = {0, -1, 0, 1};
    static constexpr int dj[] = {1, 0, -1, 0};
    int N = S.size();
    std::vector<std::vector<bool>> used(N, std::vector<bool>(N, false));
    int max_sz = 0, sum = 0;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (S[r][c] != ch || used[r][c]) continue;
            std::queue<int> qu;
            used[r][c] = true;
            qu.push(r << 6 | c);
            int sz = 1;
                while (!qu.empty()) {
                int crcc = qu.front(); qu.pop();
                int cr = crcc >> 6, cc = crcc & 0b111111;
                for (int d = 0; d < 4; d++) {
                    int nr = cr + di[d], nc = cc + dj[d];
                    if (nr < 0 || nr >= N || nc < 0 || nc >= N || used[nr][nc] || S[nr][nc] != ch) continue;
                    used[nr][nc] = true;
                    qu.push(nr << 6 | nc);
                    sz++;
                }
            }
            max_sz = std::max(max_sz, sz);
            sum += sz;
        }
    }
    return { max_sz, sum };
}

SJudgeResultPtr judge(SProblemPtr prob, SSolutionPtr sol, bool verbose) {
    SJudgeResultPtr res = std::make_shared<SJudgeResult>();

    res->N = prob->S.size();
    res->prob = prob;
    res->sol = sol;

    int N = res->N;
    auto S = prob->S;
    auto T = sol->T;

    if(verbose) {
        std::cerr << "--- problem  ---" << std::endl;
        for(const auto& s : S) {
            std::cerr << s << std::endl;
        }
        std::cerr << std::endl;
        std::cerr << "--- solution ---" << std::endl;
        for(const auto& t : T) {
            std::cerr << t << std::endl;
        }
        std::cerr << std::endl;
    }
    
    if (T.size() != N) {
        std::cerr << format("There must be %d rows, but %lld.", N, T.size()) << std::endl;
        res->score = -1;
        res->is_valid = false;
        return res;
    }

    for(int i = 0; i < N; i++) {
        const auto& t = T[i];
        if (t.size() == N) continue;
        std::cerr << format("invalid row length at line %d: %d expected, but %lld", i, N, t.size()) << std::endl;
        res->score = -1;
        res->is_valid = false;
        return res;
    }

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            // ox はそのまま
            // . は .+- のどれか
            if ((S[i][j] == 'o' || S[i][j] == 'x') && S[i][j] != T[i][j]) {
                std::cerr << format("The value of cell [%d, %d] must be the same as the input.", i, j) << std::endl;
                res->score = -1;
                res->is_valid = false;
                return res;
            }
            if (S[i][j] == '.' && (T[i][j] != '.' && T[i][j] != '+' && T[i][j] != '-')) {
                std::cerr << format("The value of cell [%d, %d] must be one of \'.\', \'+\', \'-\'.", i, j) << std::endl;
                res->score = -1;
                res->is_valid = false;
                return res;
            }
        }
    }

    auto rot_cw = [&N](const std::vector<std::string>& src) {
        std::vector<std::string> dst(N, std::string(N, ' '));
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                dst[i][j] = src[N - j - 1][i];
            }
        }
        return dst;
    };

    auto drop = [&N] (std::string& row) {
        int l = 0, r;
        while (l < N && row[l] != '.') {
            l++;
        }
        for(r = l + 1; r < N; r++) {
            if (row[r] == '-') {
                l = r + 1;
                while (l < N && row[l] != '.') {
                    l++;
                }
                r = l;
            }
            else if (row[r] != '.') {
                row[l++] = row[r], row[r] = '.';
            }
        }
    };

    T = rot_cw(T);
    for (auto& t : T) drop(t);
    for (int i = 0; i < 3; i++) T = rot_cw(T);

    auto blobs = enum_blobs(T);

    std::map<char, Blob> largest_blob;
    for(const auto& blob : blobs) {
        if(!largest_blob.count(blob.ch) || (largest_blob[blob.ch].points.size() < blob.points.size())) {
            largest_blob[blob.ch] = blob;
        }
    }

    if (verbose) {
        std::vector<std::vector<std::string>> board(N, std::vector<std::string>(N));
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                board[i][j] = std::string() + T[i][j];
            }
        }
        for(const auto& [i, j] : largest_blob['o'].points) {
            board[i][j] = CC_CYAN + board[i][j] + CC_RESET;
        }
        for(const auto& [i, j] : largest_blob['x'].points) {
            board[i][j] = CC_MAGENTA + board[i][j] + CC_RESET;
        }
        std::cerr << "--- dropped  ---" << std::endl;
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                std::cerr << board[i][j];
            }
            std::cerr << std::endl;
        }
        std::cerr << std::endl;
    }

    res->score = largest_blob['o'].points.size() + largest_blob['x'].points.size();
    res->is_valid = true;

    return res;
}



int main(int argc, char** argv) {

    CLI::App app{ "judge" };

    std::string input_path;
    std::string output_path;
    bool verbose;
    app.add_option("-i,--input", input_path, "input path")->required()->check(CLI::ExistingFile);
    app.add_option("-o,--output", output_path, "output path")->required()->check(CLI::ExistingFile);
    app.add_option("-v,--verbose", verbose, "verbose mode")->default_val(false);

    CLI11_PARSE(app, argc, argv);

    std::ifstream input_file(input_path);
    std::ifstream output_file(output_path);

    SProblemPtr prob = SProblem::load(input_file);
    SSolutionPtr sol = SSolution::load(output_file);

    auto result = judge(prob, sol, verbose);

    std::cout << format("Score = %d", result->score) << std::endl;

    return 0;
}