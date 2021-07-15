#pragma once
#include "atcoder-util.hpp"

struct SProblem;
using SProblemPtr = std::shared_ptr<SProblem>;
struct SProblem {
    int N;
    std::vector<std::vector<int>> A;
    static SProblemPtr generate(int N, int seed);
    static SProblemPtr load(std::istream& in);
    std::string str() const;
    friend std::ostream& operator<<(std::ostream& o, const SProblem& obj);
    friend std::ostream& operator<<(std::ostream& o, const SProblemPtr& obj);
};

struct SSolution;
using SSolutionPtr = std::shared_ptr<SSolution>;
struct SSolution {
    std::vector<std::pair<int, int>> moves;
    static SSolutionPtr load(std::istream& in);
};