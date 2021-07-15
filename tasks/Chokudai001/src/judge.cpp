#define _CRT_SECURE_NO_WARNINGS
#include "CLI11.hpp"
#include "atcoder-util.hpp"
#include "contest_types.hpp"
#include "json.hpp"
#include <filesystem>


struct SJudgeResult;
using SJudgeResultPtr = std::shared_ptr<SJudgeResult>;
struct SJudgeResult {
    int N;
    SProblemPtr prob;
    SSolutionPtr sol;
    int num_moves;
    std::vector<std::vector<std::pair<int, int>>> move_chains;
    int score;
    bool is_valid;
    nlohmann::json to_json() const {
        nlohmann::json json;
        json["N"] = N;
        json["num_moves"] = num_moves;
        json["score"] = score;
        json["move_chains"] = move_chains;
        json["is_valid"] = is_valid;
        return json;
    }
};

SJudgeResultPtr judge(SProblemPtr prob, SSolutionPtr sol, bool verbose) {
    SJudgeResultPtr res = std::make_shared<SJudgeResult>();

    res->prob = prob;
    res->sol = sol;

    auto board = prob->A;
    auto moves = sol->moves;
    int N = (int)board.size();

    auto print_board = [&]() {
        std::cerr << "board:" << std::endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cerr << board[i][j] << ' ';
            }
            std::cerr << std::endl;
        }
    };

    if (verbose) {
        std::cerr << format("board size: %d", N) << std::endl;
        print_board();
    }

    res->N = N;

    auto is_inside = [&N](int i, int j) { return 0 <= i && i < N && 0 <= j && j < N; };
    auto is_adjacent = [](int i1, int j1, int i2, int j2) { return abs(i1 - i2) + abs(j1 - j2) == 1; };

    res->num_moves = 0;

    auto print_verbose_info = [&]() {
        std::cerr << format("move_chain %lld: ", res->move_chains.size()) << res->move_chains.back() << std::endl;
        print_board();
    };
    
    int pi = -2, pj = -2;
    for (int row = 0; row < (int)moves.size(); row++) {
        auto [i, j] = moves[row];
        i--; j--;
        if (!is_inside(i, j)) {
            std::cerr << format("Invalid cell [%d, %d] is specfied at row %d.", i + 1, j + 1, row) << std::endl;
            res->is_valid = false;
            res->score = -1;
            return res;
        }
        if (board[i][j] == 0) {
            std::cerr << format("An operation on cell [%d, %d] with a value 0 was found in row %d", i + 1, j + 1, row) << std::endl;
            res->is_valid = false;
            res->score = -1;
            return res;
        }
        if (!is_adjacent(pi, pj, i, j) || board[pi][pj] != board[i][j]) {
            if (verbose && !res->move_chains.empty()) print_verbose_info();
            res->num_moves++;
            res->move_chains.emplace_back();
        }
        board[i][j]--;
        res->move_chains.back().emplace_back(i + 1, j + 1);
        pi = i; pj = j;
    }
    
    res->is_valid = [&]() {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (board[i][j] > 0) {
                    std::cerr << format("The value of cell [%d, %d] is greater than 0.", i + 1, j + 1) << std::endl;
                    return false;
                }
            }
        }
        return true;
    }();
    if (!res->is_valid) {
        res->score = -1;
        return res;
    }

    if (verbose) print_verbose_info();

    res->score = 100000 - res->num_moves;
    res->is_valid = true;
    return res;
}



int main(int argc, char** argv) {

    CLI::App app{ "judge" };

    std::string input_path;
    std::string output_path;
    std::string result_path;
    bool verbose;
    app.add_option("-i,--input", input_path, "input path")->required()->check(CLI::ExistingFile);
    app.add_option("-o,--output", output_path, "output path")->required()->check(CLI::ExistingFile);
    app.add_option("-r,--result", result_path, "result path")->check(CLI::NonexistentPath);
    app.add_option("-v,--verbose", verbose, "verbose mode")->default_val(false);

    CLI11_PARSE(app, argc, argv);

    std::ifstream input_file(input_path);
    std::ifstream output_file(output_path);

    SProblemPtr prob = SProblem::load(input_file);
    SSolutionPtr sol = SSolution::load(output_file);

    auto result = judge(prob, sol, verbose);

    std::cout << format("Score = %d", result->score) << std::endl;

    if (!result_path.empty()) {
        std::ofstream result_file(result_path);
        result_file << result->to_json();
        result_file.close();
    }

    return 0;
}