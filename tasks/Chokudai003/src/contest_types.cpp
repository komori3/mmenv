#include "contest_types.hpp"

SProblemPtr SProblem::generate(int N, int seed) {
    SProblemPtr p = std::make_shared<SProblem>();
    Xorshift rnd(seed);
    p->N = N;
    p->S.resize(N, std::string(N, '.'));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int r = rnd.next_int(100);
            if (r < 25) p->S[i][j] = 'o';
            else if (r < 50) p->S[i][j] = 'x';
        }
    }
    return p;
}

SProblemPtr SProblem::load(std::istream& in) {
    SProblemPtr p = std::make_shared<SProblem>();
    std::string buf;
    while (in >> buf) {
        if (buf.empty()) continue;
        p->S.push_back(buf);
    }
    p->N = p->S.size();
    return p;
}

std::string SProblem::str() const {
    std::ostringstream oss;
    for (int i = 0; i < N; i++) {
        oss << S[i] << '\n';
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
    std::string buf;
    while (in >> buf) {
        if(buf.empty()) continue;
        s->T.push_back(buf);
    }
    return s;
}