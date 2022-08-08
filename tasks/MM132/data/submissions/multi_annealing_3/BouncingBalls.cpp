#include <bits/stdc++.h>
#include <random>
#ifdef _MSC_VER
#include <ppl.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

/** compro_io **/

/* tuple */
// out
namespace aux {
    template<typename T, unsigned N, unsigned L>
    struct tp {
        static void output(std::ostream& os, const T& v) {
            os << std::get<N>(v) << ", ";
            tp<T, N + 1, L>::output(os, v);
        }
    };
    template<typename T, unsigned N>
    struct tp<T, N, N> {
        static void output(std::ostream& os, const T& v) { os << std::get<N>(v); }
    };
}
template<typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) {
    os << '[';
    aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t);
    return os << ']';
}

template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x);

/* pair */
// out
template<class S, class T>
std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) {
    return os << "[" << p.first << ", " << p.second << "]";
}
// in
template<class S, class T>
std::istream& operator>>(std::istream& is, const std::pair<S, T>& p) {
    return is >> p.first >> p.second;
}

/* container */
// out
template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) {
    bool f = true;
    os << "[";
    for (auto& y : x) {
        os << (f ? "" : ", ") << y;
        f = false;
    }
    return os << "]";
}
// in
template <
    class T,
    class = decltype(std::begin(std::declval<T&>())),
    class = typename std::enable_if<!std::is_same<T, std::string>::value>::type
>
std::istream& operator>>(std::istream& is, T& a) {
    for (auto& x : a) is >> x;
    return is;
}

/* struct */
template<typename T>
auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) {
    out << t.stringify();
    return out;
}

/* setup */
struct IOSetup {
    IOSetup(bool f) {
        if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); }
        std::cout << std::fixed << std::setprecision(15);
    }
} iosetup(true);

/** string formatter **/
template<typename... Ts>
std::string format(const std::string& f, Ts... t) {
    size_t l = std::snprintf(nullptr, 0, f.c_str(), t...);
    std::vector<char> b(l + 1);
    std::snprintf(&b[0], l + 1, f.c_str(), t...);
    return std::string(&b[0], &b[0] + l);
}

template<typename T>
std::string stringify(const T& x) {
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

/* dump */
#define ENABLE_DUMP
#ifdef ENABLE_DUMP
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }
#else
#define dump(...) void(0);
#endif

/* timer */
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 2.8e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 2.8e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() { return (time() - t - paused) * 1000.0; }
} timer;

/* rand */
struct Xorshift {
    uint64_t x = 88172645463325252LL;
    void set_seed(unsigned seed, int rep = 100) { x = uint64_t((seed + 1) * 10007); for (int i = 0; i < rep; i++) next_int(); }
    unsigned next_int() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    unsigned next_int(unsigned mod) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % mod; }
    unsigned next_int(unsigned l, unsigned r) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % (r - l + 1) + l; } // inclusive
    double next_double() { return double(next_int()) / UINT_MAX; }
} rnd;

/* shuffle */
template<typename T>
void shuffle_vector(std::vector<T>& v, Xorshift& rnd) {
    int n = v.size();
    for (int i = n - 1; i >= 1; i--) {
        int r = rnd.next_int(i);
        std::swap(v[i], v[r]);
    }
}

/* split */
std::vector<std::string> split(std::string str, const std::string& delim) {
    for (char& c : str) if (delim.find(c) != std::string::npos) c = ' ';
    std::istringstream iss(str);
    std::vector<std::string> parsed;
    std::string buf;
    while (iss >> buf) parsed.push_back(buf);
    return parsed;
}

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) {
    std::fill((T*)array, (T*)(array + N), val);
}

template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }


using pii = std::pair<int, int>;
constexpr int dr[] = { 0, -1, 0, 1 };
constexpr int dc[] = { 1, 0, -1, 0 };



namespace NInput {

    int N;
    int B;
    int bonus_val;
    std::array<std::array<char, 32>, 32> grid;

    void init(std::istream& in) {
        in >> N >> B >> bonus_val;
        for (int r = 0; r < 32; r++) {
            for (int c = 0; c < 32; c++) {
                grid[r][c] = '#';
            }
        }
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                in >> grid[r][c];
            }
        }
    }

}

struct MoveBase;
using MoveBasePtr = std::shared_ptr<MoveBase>;
struct MoveBase {
    enum struct Type {
        GUN, GRID
    };
    Type type;
    MoveBase(Type type) : type(type) {}
};

struct MoveGun;
using MoveGunPtr = std::shared_ptr<MoveGun>;
struct MoveGun : MoveBase {
    int id, r1, c1, r2, c2;
    MoveGun(int id, int r1, int c1, int r2, int c2) : MoveBase(Type::GUN), id(id), r1(r1), c1(c1), r2(r2), c2(c2) {}
};

struct MoveGrid;
using MoveGridPtr = std::shared_ptr<MoveGrid>;
struct MoveGrid : MoveBase {
    int r, c;
    char from, to;
    MoveGrid(int r, int c, char from, char to) : MoveBase(Type::GRID), r(r), c(c), from(from), to(to) {}
};

struct State {

    std::array<int, 5> m_gr;
    std::array<int, 5> m_gc;
    std::array<std::array<char, 32>, 32> m_grid;

    int evaluate() const {
        auto gr(m_gr);
        auto gc(m_gc);
        auto grid(m_grid);
        const int N = NInput::N;
        const int B = NInput::B;
        const int bonus_val = NInput::bonus_val;

        auto get_dir = [&](int r, int c) {
            if (c == 0) return 0;
            if (r == N + 1) return 1;
            if (c == N + 1) return 2;
            return 3;
        };

        std::array<int, 5> dirs;
        for (int i = 0; i < B; i++) {
            dirs[i] = get_dir(gr[i], gc[i]);
        }

        int score = 0;
        std::array<std::array<int, 32>, 32> num_balls;
        std::memset(num_balls.data(), 0, sizeof(int) * 1024);
        for (int i = 0; i < B; i++) num_balls[gr[i]][gc[i]]++;

        while (true) {
            bool end_flag = false;
            // 弾の移動
            for (int i = 0; i < B; i++) {
                int& r = gr[i];
                int& c = gc[i];

                num_balls[r][c]--;
                r += dr[dirs[i]]; c += dc[dirs[i]];
                num_balls[r][c]++;

                if (grid[r][c] == '*') score += bonus_val;
                else if (grid[r][c] == '/' || grid[r][c] == '\\') score++;

                if (dirs[i] == 0) {
                    if (grid[r][c] == '\\') dirs[i] = 3;
                    else if (grid[r][c] == '/') dirs[i] = 1;
                }
                else if (dirs[i] == 1) {
                    if (grid[r][c] == '\\') dirs[i] = 2;
                    else if (grid[r][c] == '/') dirs[i] = 0;
                }
                else if (dirs[i] == 2) {
                    if (grid[r][c] == '\\') dirs[i] = 1;
                    else if (grid[r][c] == '/') dirs[i] = 3;
                }
                else {
                    if (grid[r][c] == '\\') dirs[i] = 0;
                    else if (grid[r][c] == '/') dirs[i] = 2;
                }

                if (grid[r][c] == '#') end_flag = true;
            }
            // 反射板の変化
            for (int i = 0; i < B; i++) {
                int r = gr[i], c = gc[i];
                if (num_balls[r][c] == 1 && NInput::grid[r][c] == '.' && grid[r][c] != '.') {
                    //assert(grid[r][c] == '/' || grid[r][c] == '\\');
                    if (grid[r][c] == '/') grid[r][c] = '\\';
                    else grid[r][c] = '/';
                }
            }

            if (end_flag) break;
        }

        return score;
    }

    void output(std::ostream& out) const {
        for (int i = 0; i < NInput::B; i++) {
            out << m_gr[i] - 1 << ' ' << m_gc[i] - 1 << '\n';
        }
        for (int r = 1; r <= NInput::N; r++) {
            for (int c = 1; c <= NInput::N; c++) {
                out << m_grid[r][c] << '\n';
            }
        }
        out.flush();
    }

    static State generate_random(Xorshift& rnd) {
        int N = NInput::N;
        int B = NInput::B;

        std::set<pii> seen;
        std::vector<pii> guns;
        auto grid(NInput::grid);

        // gun locations
        while ((int)guns.size() < B) {
            int type = rnd.next_int(4);
            int loc = rnd.next_int(N) + 1;
            pii p;
            if (type == 0) p = pii(0, loc);
            else if (type == 1) p = pii(N + 1, loc);
            else if (type == 2) p = pii(loc, 0);
            else p = pii(loc, N + 1);
            if (seen.count(p)) continue;
            seen.insert(p);
            guns.push_back(p);
        }

        char cells[] = { '/','\\','.' };
        // grid
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++)
            {
                if (NInput::grid[r + 1][c + 1] == '.')
                {
                    int type = rnd.next_int(2);
                    //int type = 0;
                    grid[r + 1][c + 1] = cells[type];
                }
            }
        }

        State state;
        for (int i = 0; i < NInput::B; i++) {
            state.m_gr[i] = guns[i].first;
            state.m_gc[i] = guns[i].second;
        }
        state.m_grid = grid;

        return state;
    }

    void move(MoveGunPtr mv) {
        m_gr[mv->id] = mv->r2;
        m_gc[mv->id] = mv->c2;
    }

    void move(MoveGridPtr mv) {
        m_grid[mv->r][mv->c] = mv->to;
    }

    void move(MoveBasePtr mv) {
        if (mv->type == MoveBase::Type::GUN) {
            move(std::reinterpret_pointer_cast<MoveGun>(mv));
        }
        else {
            move(std::reinterpret_pointer_cast<MoveGrid>(mv));
        }
    }

    void undo(MoveGunPtr move) {
        m_gr[move->id] = move->r1;
        m_gc[move->id] = move->c1;
    }

    void undo(MoveGridPtr move) {
        m_grid[move->r][move->c] = move->from;
    }

    void undo(MoveBasePtr move) {
        if (move->type == MoveBase::Type::GUN) {
            undo(std::reinterpret_pointer_cast<MoveGun>(move));
        }
        else {
            undo(std::reinterpret_pointer_cast<MoveGrid>(move));
        }
    }

    MoveBasePtr random_move(Xorshift& rnd) {
        int N = NInput::N, B = NInput::B;
        if (rnd.next_int(2)) {
            // gun
            std::set<pii> seen;
            for (int i = 0; i < B; i++) seen.emplace(m_gr[i], m_gc[i]);
            int id = rnd.next_int(B);
            while (true) {
                int type = rnd.next_int(4);
                int loc = rnd.next_int(N) + 1;
                pii p;
                if (type == 0) p = pii(0, loc);
                else if (type == 1) p = pii(N + 1, loc);
                else if (type == 2) p = pii(loc, 0);
                else p = pii(loc, N + 1);
                if (seen.count(p)) continue;
                auto m = std::make_shared<MoveGun>(id, m_gr[id], m_gc[id], p.first, p.second);
                move(m);
                return m;
            }
        }
        else {
            // grid
            while (true) {
                int i = rnd.next_int(N) + 1, j = rnd.next_int(N) + 1;
                if (NInput::grid[i][j] != '.') continue;
                if (m_grid[i][j] == '/') {
                    auto m = std::make_shared<MoveGrid>(i, j, '/', '\\');
                    move(m);
                    return m;
                }
                else if (m_grid[i][j] == '\\') {
                    auto m = std::make_shared<MoveGrid>(i, j, '\\', '/');
                    move(m);
                    return m;
                }
            }
        }
        return nullptr;
    }

};

MoveBasePtr calc_greedy_move(const State& state) {
    static constexpr char type[] = "/\\.";
    int N = NInput::N, B = NInput::B;
    int best_score = state.evaluate();
    int best_r = -1, best_c = -1;
    char best_pc = '$', best_nc = '$';
    for (int r = 1; r <= N; r++) {
        for (int c = 1; c <= N; c++) {
            if (NInput::grid[r][c] != '.') continue;
            char pc = state.m_grid[r][c];
            for (int t = 0; t < 2; t++) {
                char nc = type[t];
                if (pc == type[t]) continue;
                State new_state(state);
                new_state.m_grid[r][c] = nc;
                int score = new_state.evaluate();
                if (best_score < score) {
                    best_score = score;
                    best_r = r;
                    best_c = c;
                    best_pc = pc;
                    best_nc = nc;
                }
            }
        }
    }
    return best_r == -1 ? nullptr : std::make_shared<MoveGrid>(best_r, best_c, best_pc, best_nc);
}


struct MultiSA {

    const int num_state;
    const double max_temp;
    const double min_temp;

    std::vector<bool> alive;
    std::vector<State> states;
    std::vector<int> scores;

    State best_state;
    int best_score;

    MultiSA(int num_state, double max_temp, double min_temp) : num_state(num_state), min_temp(min_temp), max_temp(max_temp) {
        alive.resize(num_state, true);
        for (int i = 0; i < num_state; i++) {
            states.push_back(State::generate_random(rnd));
            scores.push_back(states.back().evaluate());
        }
        best_score = INT_MIN;
    }

    void run(double duration) {

        auto get_temp = [](double start_temp, double end_temp, double progress) {
            return end_temp + (start_temp - end_temp) * (1.0 - progress);
        };

        double start_time = timer.elapsed_ms(), now_time, end_time = start_time + duration;
        while ((now_time = timer.elapsed_ms()) < end_time) {
            double temp = get_temp(max_temp, min_temp, (now_time - start_time) / (end_time - start_time));
            for (int i = 0; i < states.size(); i++) if(alive[i]) {
                State& state = states[i];
                int& score = scores[i];
                auto mv = state.random_move(rnd);
                int new_score = state.evaluate();
                int diff = new_score - score;
                double prob = std::exp(diff / temp);
                if (prob > rnd.next_double()) {
                    score = new_score;
                    if (best_score < score) {
                        best_score = score;
                        best_state = state;
                        dump(i, best_score);
                    }
                }
                else {
                    state.undo(mv);
                }
            }
        }

    }

};


int main() {

#if 0
    std::ifstream ifs("C:\\Users\\komori3\\OneDrive\\dev\\heuristic\\tasks\\MM132\\in\\2.in");
    std::istream& in = ifs;
    std::ofstream ofs("C:\\Users\\komori3\\OneDrive\\dev\\heuristic\\tasks\\MM132\\out\\2.out");
    std::ostream& out = ofs;
#else
    std::istream& in = std::cin;
    std::ostream& out = std::cout;
#endif

    NInput::init(in);
    dump(NInput::N, NInput::B, NInput::bonus_val);

    MultiSA re(10, 100, 0);
    re.run(5000);

    //auto state = State::generate_random(rnd);
    auto state = re.best_state;
    int score = state.evaluate();
    auto best_state(state);
    auto best_score(score);
    dump(score);

    auto get_temp = [](double start_temp, double end_temp, double progress) {
        return end_temp + (start_temp - end_temp) * (1.0 - progress);
    };

    int loop = 0;
    double start_time = timer.elapsed_ms(), now_time, end_time = 9000;
    while ((now_time = timer.elapsed_ms()) < end_time) {
        loop++;
        auto mv = state.random_move(rnd);
        int new_score = state.evaluate();
        int diff = new_score - score;
        double temp = get_temp(10, 0, (now_time - start_time) / (end_time - start_time));
        double prob = std::exp(diff / temp);
        if (prob > rnd.next_double()) {
            score = new_score;
            if (best_score < score) {
                best_score = score;
                best_state = state;
                dump(loop, best_score);
            }
        }
        else {
            state.undo(mv);
        }
    }
    dump(loop, best_score);
    best_state.output(out);

    return 0;
}