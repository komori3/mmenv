#pragma once
#include "atcoder-util.hpp"
#include "contest_types.hpp"

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
};

SJudgeResultPtr judge(SProblemPtr prob, SSolutionPtr sol, bool verbose = false);