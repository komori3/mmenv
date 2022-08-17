#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#ifdef _MSC_VER
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <conio.h>
#include <ppl.h>
#include <filesystem>
#include <intrin.h>
//#include <boost/multiprecision/cpp_int.hpp>
int __builtin_clz(unsigned int n)
{
    unsigned long index;
    _BitScanReverse(&index, n);
    return 31 - index;
}
int __builtin_ctz(unsigned int n)
{
    unsigned long index;
    _BitScanForward(&index, n);
    return index;
}
namespace std {
    inline int __lg(int __n) { return sizeof(int) * 8 - 1 - __builtin_clz(__n); }
}
//using __uint128_t = boost::multiprecision::uint128_t;
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
std::istream& operator>>(std::istream& is, std::pair<S, T>& p) {
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

std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) {
    std::string s(v.size(), ' ');
    for (int i = 0; i < v.size(); i++) s[i] = v[i] + '0';
    os << s;
    return os;
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
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }

/* timer */
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 3.0e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 3.0e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
};

/* rand */
struct Xorshift {
    static constexpr uint64_t M = INT_MAX;
    static constexpr double e = 1.0 / M;
    uint64_t x = 88172645463325252LL;
    Xorshift() {}
    Xorshift(uint64_t seed) { reseed(seed); }
    inline void reseed(uint64_t seed) { x = 0x498b3bc5 ^ seed; for (int i = 0; i < 20; i++) next(); }
    inline uint64_t next() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    inline int next_int() { return next() & M; }
    inline int next_int(int mod) { return next() % mod; }
    inline int next_int(int l, int r) { return l + next_int(r - l + 1); }
    inline double next_double() { return next_int() * e; }
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

template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }

using ll = long long;
using ld = double;
//using ld = boost::multiprecision::cpp_bin_float_quad;
using pii = std::pair<int, int>;
using pll = std::pair<ll, ll>;

using std::cin, std::cout, std::cerr, std::endl, std::string, std::vector, std::array;



constexpr int NN = 50;
constexpr int KK = 500;
template<typename T> using Grid = array<array<T, NN>, NN>;

static constexpr char USED = -1;
static constexpr int di[4] = { 0, 1, 0, -1 };
static constexpr int dj[4] = { 1, 0, -1, 0 };

struct Input;
using InputPtr = std::shared_ptr<Input>;
struct Input {
    int N, K;
    Grid<char> grid;
    Input(std::istream& in) {
        in >> N >> K;
        memset(grid.data(), -1, sizeof(char) * NN * NN);
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                in >> grid[i][j];
                grid[i][j] -= '0';
            }
        }
    }
    string stringify() const {
        string res = format("%d %d\n", N, K);
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                res += char(grid[i][j] + '0');
            }
            res += '\n';
        }
        return res;
    }
};

struct MoveAction {
    int before_row, before_col, after_row, after_col;
    MoveAction(int before_row, int before_col, int after_row, int after_col) :
        before_row(before_row), before_col(before_col), after_row(after_row), after_col(after_col) {}
};

struct ConnectAction {
    int c1_row, c1_col, c2_row, c2_col;
    ConnectAction(int c1_row, int c1_col, int c2_row, int c2_col) :
        c1_row(c1_row), c1_col(c1_col), c2_row(c2_row), c2_col(c2_col) {}
};

struct Result {
    vector<MoveAction> move;
    vector<ConnectAction> connect;
    Result(const vector<MoveAction>& move = {}, const vector<ConnectAction>& con = {}) : move(move), connect(con) {}
};

struct UnionFind {
    vector<int> data;

    UnionFind() = default;

    explicit UnionFind(size_t sz) : data(sz, -1) {}

    bool unite(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) return false;
        if (data[x] > data[y]) std::swap(x, y);
        data[x] += data[y];
        data[y] = x;
        return true;
    }

    int find(int k) {
        if (data[k] < 0) return (k);
        return data[k] = find(data[k]);
    }

    int size(int k) {
        return -data[find(k)];
    }

    bool same(int x, int y) {
        return find(x) == find(y);
    }

    vector<vector<int>> groups() {
        int n = (int)data.size();
        vector< vector< int > > ret(n);
        for (int i = 0; i < n; i++) {
            ret[find(i)].emplace_back(i);
        }
        ret.erase(remove_if(begin(ret), end(ret), [&](const vector< int >& v) {
            return v.empty();
            }), ret.end());
        return ret;
    }
};

struct Cell {
    int id;
    int i, j;
    int color;
    Cell(int id = -1, int i = -1, int j = -1, int color = -1) : id(id), i(i), j(j), color(color) {}
    string stringify() const {
        return format("Cell [id=%d, i=%d, j=%d, color=%d]", id, i, j, color);
    }
};

struct State {

    static constexpr short WALL = -1;
    static constexpr short EMPTY = 0;

    int N, K;
    int V;
    Cell cells[KK + 1]; // 1-indexed
    Grid<short> grid; // id
    short nexts[KK + 1][4];

    int M;
    int moves[KK + 1];

    State() {}

    State(int N, int K, const Grid<char>& g) : N(N), K(K), V(0), M(0) {
        memset(cells, -1, sizeof(Cell) * (KK + 1));
        memset(grid.data(), -1, sizeof(short) * NN * NN);
        memset(nexts, -1, sizeof(short) * (KK + 1) * 4);
        memset(moves, -1, sizeof(std::pair<short, short>) * (KK * 1));
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                if (g[i][j]) {
                    V++;
                    cells[V] = Cell(V, i, j, g[i][j]);
                    grid[i][j] = V;
                }
                else {
                    grid[i][j] = EMPTY;
                }
            }
        }
        update();
    }

    void update() {
        for (int i = 1; i <= V; i++) {
            for (int d = 0; d < 4; d++) {
                nexts[cells[i].id][d] = search(cells[i], d);
            }
        }
    }

    inline short search(int i, int j, int d) const {
        i += di[d]; j += dj[d];
        while (!grid[i][j]) i += di[d], j += dj[d];
        return grid[i][j];
    }

    inline short search(const Cell& c, int d) const {
        return search(c.i, c.j, d);
    }

    bool can_move(int id, int d) const {
        return !grid[cells[id].i + di[d]][cells[id].j + dj[d]];
    }

    inline int pack(int i1, int j1, int i2, int j2) const {
        return (i1 << 24) | (j1 << 16) | (i2 << 8) | j2;
    }

    inline std::tuple<int, int, int, int> unpack(int p) const {
        return {
            (p >> 24) & 0xFF,
            (p >> 16) & 0xFF,
            (p >> 8) & 0xFF,
            p & 0xFF
        };
    }

    void move(int id, int d) {
        auto& cell = cells[id];
        moves[M++] = pack(cell.i, cell.j, cell.i + di[d], cell.j + dj[d]);
        // (d+1)&3, (d+3)&3 �������X�V
        int d1 = (d + 1) & 3, d2 = (d + 3) & 3;
        // erase
        int id1 = nexts[id][d1], id2 = nexts[id][d2];
        if (id1 != WALL) nexts[id1][d2] = id2;
        if (id2 != WALL) nexts[id2][d1] = id1;
        grid[cell.i][cell.j] = EMPTY;
        // insert
        cell.i += di[d]; cell.j += dj[d];
        grid[cell.i][cell.j] = id;
        id1 = nexts[id][d1] = search(cell.i, cell.j, d1);
        id2 = nexts[id][d2] = search(cell.i, cell.j, d2);
        if (id1 != WALL) nexts[id1][d2] = id;
        if (id2 != WALL) nexts[id2][d1] = id;
    }

    vector<pii> enum_moves() const {
        vector<pii> res;
        for (int id = 1; id <= V; id++) {
            for (int d = 0; d < 4; d++) {
                if (can_move(id, d)) res.emplace_back(id, d);
            }
        }
        return res;
    }

    int eval() const {
#if 0
        bool used[KK + 1] = {};
        int score = 0;
        for (int s = 1; s <= V; s++) {
            const auto& sc = cells[s];
            if (used[s]) continue;
            int nc = 0;
            std::queue<int> qu({ s });
            used[s] = true;
            nc++;
            while (!qu.empty()) {
                auto u = qu.front(); qu.pop();
                for (int v : nexts[u]) {
                    if (v != -1 && !used[v] && cells[u].color == cells[v].color) {
                        qu.push(v);
                        used[v] = true;
                        nc++;
                    }
                }
            }
            score += nc * (nc - 1) / 2;
        }
#else
        int score = 0;
        for (int u = 1; u <= V; u++) {
            int ucol = cells[u].color;
            for (int v : nexts[u]) {
                if (v == -1) continue;
                score += ucol == cells[v].color;
            }
        }
#endif
        return score;
    }

    vector<int> get_max_cluster() const {
        bool used[KK + 1] = {};
        int cluster_id[KK + 1] = {};
        int id = 0, max_id = -1, max_nc = -1;
        for (int s = 1; s <= V; s++) {
            const auto& sc = cells[s];
            if (used[s]) continue;
            id++;
            int nc = 1;
            std::queue<int> qu({ s });
            used[s] = true;
            cluster_id[s] = id;
            while (!qu.empty()) {
                auto u = qu.front(); qu.pop();
                for (int v : nexts[u]) {
                    if (v != -1 && !used[v] && cells[u].color == cells[v].color) {
                        qu.push(v);
                        used[v] = true;
                        cluster_id[v] = id;
                        nc++;
                    }
                }
            }
            if (chmax(max_nc, nc)) {
                max_id = id;
            }
        }
        vector<int> res;
        for (int i = 1; i <= V; i++) if (cluster_id[i] == max_id) res.push_back(i);
        return res;
    }

    inline int get_dir(int i1, int j1, int i2, int j2) const {
        if (j1 < j2) return 0;
        if (i1 < i2) return 1;
        if (j2 < j1) return 2;
        return 3;
    }

    void connect(const Cell& c1, const Cell& c2) {
        int d = get_dir(c1.i, c1.j, c2.i, c2.j);
        int i = c1.i, j = c1.j;
        while (i != c2.i || j != c2.j) {
            grid[i][j] = WALL;
            i += di[d]; j += dj[d];
        }
        grid[i][j] = WALL;
    }

    int greedy_connect(int s, vector<ConnectAction>& conn, int& rem) {
        std::queue<int> qu({ s });
        int nc = 1;
        while (!qu.empty() && rem) {
            int u = qu.front(); qu.pop();
            const auto& uc = cells[u];
            for (int v : nexts[u]) {
                if (!rem || v == -1) continue;
                const auto& vc = cells[v];
                if (grid[vc.i][vc.j] == WALL || uc.color != vc.color) continue;
                qu.push(v);
                connect(uc, vc);
                conn.emplace_back(uc.i, uc.j, vc.i, vc.j);
                update();
                rem--;
                nc++;
            }
        }
        return nc;
    }

    std::pair<int, Result> post_process() {
        vector<MoveAction> mvs;
        for (int i = 0; i < M; i++) {
            auto [i1, j1, i2, j2] = unpack(moves[i]);
            mvs.emplace_back(i1, j1, i2, j2);
        }
        vector<ConnectAction> conn;
        int rem = K * 100 - M, score = 0;
        while (rem) {
            auto cs = get_max_cluster();
            if (cs.size() == 1) break;
            int nc = greedy_connect(cs.front(), conn, rem);
            score += nc * (nc - 1) / 2;
        }
        return { score, { mvs, conn } };
    }

    void print() const {
        for (int i = 0; i <= N + 1; i++) {
            for (int j = 0; j <= N + 1; j++) {
                if (grid[i][j] == WALL) fprintf(stderr, "#");
                else if (grid[i][j]) fprintf(stderr, "%d", cells[grid[i][j]].color);
                else fprintf(stderr, " ");
            }
            fprintf(stderr, "\n");
        }
    }

};

int calc_score(InputPtr input, const Result& res) {

    struct UnionFind_ {
        std::map<pii, pii> parent;
        UnionFind_() :parent() {}

        pii find(pii x)
        {
            if (parent.find(x) == parent.end()) {
                parent[x] = x;
                return x;
            }
            else if (parent[x] == x) {
                return x;
            }
            else {
                parent[x] = find(parent[x]);
                return parent[x];
            }
        }

        void unite(pii x, pii y)
        {
            x = find(x);
            y = find(y);
            if (x != y) {
                parent[x] = y;
            }
        }
    };

    auto N = input->N;
    auto field = input->grid;
    for (auto r : res.move) {
        assert(field[r.before_row][r.before_col]);
        assert(!field[r.after_row][r.after_col]);
        std::swap(field[r.before_row][r.before_col], field[r.after_row][r.after_col]);
    }

    UnionFind_ uf;
    for (auto r : res.connect) {
        pii p1(r.c1_row, r.c1_col), p2(r.c2_row, r.c2_col);
        uf.unite(p1, p2);
    }

    vector<pii> computers;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            if (field[i][j]) {
                computers.emplace_back(i, j);
            }
        }
    }

    int score = 0;
    for (int i = 0; i < (int)computers.size(); i++) {
        for (int j = i + 1; j < (int)computers.size(); j++) {
            auto c1 = computers[i];
            auto c2 = computers[j];
            if (uf.find(c1) == uf.find(c2)) {
                score += (field[c1.first][c1.second] == field[c2.first][c2.second]) ? 1 : -1;
            }
        }
    }

    return std::max(score, 0);
}

struct Solver;
using SolverPtr = std::shared_ptr<Solver>;
struct Solver {

    Timer timer;
    Xorshift rnd;

    InputPtr input;

    int N, K;
    int action_count_limit;
    Grid<char> grid;

    Solver(InputPtr input) : input(input), N(input->N), K(input->K), action_count_limit(K * 100), grid(input->grid) {}

    vector<State> beam_search(State init_state) {
        constexpr int beam_width = 3, degree = 100;
        State sbuf[2][beam_width * degree];
        int ord[beam_width * degree];
        int scores[beam_width * degree];

        int now_buffer = 0;
        int buf_size[2] = {};

        sbuf[now_buffer][0] = init_state;
        ord[0] = 0;
        buf_size[now_buffer]++;

        vector<State> res({ init_state });

        int turn = 0;
        while (buf_size[now_buffer] && turn < K * 100) {
            auto& now_states = sbuf[now_buffer];
            auto& now_size = sbuf[now_buffer];
            auto& next_states = sbuf[now_buffer ^ 1];
            auto& next_size = buf_size[now_buffer ^ 1]; next_size = 0;

            for (int n = 0; n < std::min(beam_width, buf_size[now_buffer]); n++) {
                auto& now_state = now_states[ord[n]];
                auto cands = now_state.enum_moves();
                shuffle_vector(cands, rnd);
                for (int i = 0; i < std::min(degree, (int)cands.size()); i++) {
                    auto [id, d] = cands[i];
                    auto& next_state = next_states[next_size];
                    next_state = now_state;
                    next_state.move(id, d);
                    scores[next_size] = next_state.eval();
                    next_size++;
                }
            }

            if (!next_size) break;
            std::iota(ord, ord + next_size, 0);
            std::sort(ord, ord + next_size, [&scores](int a, int b) {
                return scores[a] > scores[b];
                });

            //dump(next_states[ord[0]].eval());

            res.push_back(next_states[ord[0]]);

            now_buffer ^= 1;
            turn++;
        }
        //dump(turn, best_score);
        return res;
    }

    Result solve() {
        State state(N, K, grid);
        auto states = beam_search(state);
        Result best;
        int best_score = -1;
        for (int i = 0; i < states.size(); i += 2) {
            auto state = states[i];
            auto [score, res] = state.post_process();
            if (chmax(best_score, score)) {
                best = res;
            }
        }
        return best;
    }

};

void print_answer(std::ostream& out, const Result& res) {
    out << res.move.size() << endl;
    for (auto m : res.move) {
        out << m.before_row - 1 << " " << m.before_col - 1 << " "
            << m.after_row - 1 << " " << m.after_col - 1 << endl;
    }
    out << res.connect.size() << endl;
    for (auto m : res.connect) {
        out << m.c1_row - 1 << " " << m.c1_col - 1 << " "
            << m.c2_row - 1 << " " << m.c2_col - 1 << endl;
    }
}



#ifdef _MSC_VER
void batch_test(int seed_begin = 0, int num_seed = 100) {

    constexpr int batch_size = 8;
    int seed_end = seed_begin + num_seed;

    vector<int> scores(num_seed);
#if 1
    concurrency::critical_section mtx;
    for (int batch_begin = seed_begin; batch_begin < seed_end; batch_begin += batch_size) {
        int batch_end = std::min(batch_begin + batch_size, seed_end);
        concurrency::parallel_for(batch_begin, batch_end, [&mtx, &scores](int seed) {
            std::ifstream ifs(format("tools/in/%04d.txt", seed));
            std::istream& in = ifs;
            std::ofstream ofs(format("tools/out/%04d.txt", seed));
            std::ostream& out = ofs;

            auto input = std::make_shared<Input>(in);
            Solver solver(input);
            auto ret = solver.solve();
            print_answer(out, ret);

            {
                mtx.lock();
                scores[seed] = calc_score(input, ret);
                cerr << seed << ": " << scores[seed] << ", " << solver.timer.elapsed_ms() << '\n';
                mtx.unlock();
            }
            });
    }
#else
    for (int seed = seed_begin; seed < seed_begin + num_seed; seed++) {
        std::ifstream ifs(format("tools/in/%04d.txt", seed));
        std::istream& in = ifs;
        std::ofstream ofs(format("tools/out/%04d.txt", seed));
        std::ostream& out = ofs;

        auto input = std::make_shared<Input>(in);
        Solver solver(input);
        auto ret = solver.solve();
        print_answer(out, ret);

        scores[seed] = calc_score(input, ret);
        cerr << seed << ": " << scores[seed] << ", " << solver.timer.elapsed_ms() << '\n';
    }
#endif

    dump(std::accumulate(scores.begin(), scores.end(), 0));
}
#endif


int main(int argc, char** argv) {

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#ifdef _MSC_VER
    std::ifstream ifs(R"(tools\in\0003.txt)");
    std::ofstream ofs(R"(tools\out\0003.txt)");
    std::istream& in = ifs;
    std::ostream& out = ofs;
#else
    std::istream& in = cin;
    std::ostream& out = cout;
#endif

#if 0
    batch_test();
#else
    auto input = std::make_shared<Input>(in);
    Solver solver(input);
    auto ret = solver.solve();
    dump(calc_score(input, ret));
    print_answer(out, ret);
#endif

    return 0;
}