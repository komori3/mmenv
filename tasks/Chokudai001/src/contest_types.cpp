#include "contest_types.hpp"

SProblemPtr SProblem::generate(int N, int seed) {
    SProblemPtr p = std::make_shared<SProblem>();
    Xorshift rnd(seed);
    p->N = N;
    p->A.resize(N, std::vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            p->A[i][j] = rnd.next_int(100) + 1;
        }
    }
    return p;
}

SProblemPtr SProblem::load(std::istream& in) {
    SProblemPtr p = std::make_shared<SProblem>();
    std::vector<int> a;
    std::string buf;
    while (in >> buf) {
        a.push_back(stoi(buf));
    }
    int N;
    for (N = 1;; N++) if (N * N == a.size()) break;
    p->N = N;
    p->A.resize(N, std::vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            p->A[i][j] = a[i * N + j];
        }
    }
    return p;
}

std::string SProblem::str() const {
    std::ostringstream oss;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            oss << A[i][j] << ' ';
        }
        oss << '\n';
    }
    return oss.str();
}

std::ostream& operator<<(std::ostream& o, const SProblem& obj) {
    o << obj.str();
    return o;
}

std::ostream& operator<<(std::ostream& o, const SProblemPtr& obj) {
    o << obj->str();
    return o;
}

SSolutionPtr SSolution::load(std::istream& in) {
    SSolutionPtr s = std::make_shared<SSolution>();
    std::vector<int> a;
    std::string buf;
    while (in >> buf) {
        a.push_back(stoi(buf));
    }
    int num_moves = (int)a.size() / 2;
    for (int i = 0; i < num_moves; i++) {
        s->moves.emplace_back(a[2 * i], a[2 * i + 1]);
    }
    return s;
}