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

inline int get_dir(int i1, int j1, int i2, int j2) {
    if (j1 < j2) return 0;
    if (i1 < i2) return 1;
    if (j2 < j1) return 2;
    return 3;
}

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
                if (g[i][j] > 0) {
                    V++;
                    cells[V] = Cell(V, i, j, g[i][j]);
                    grid[i][j] = V;
                }
                else {
                    grid[i][j] = g[i][j];
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
        // (d+1)&3, (d+3)&3 方向を更新
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


template< class T >
struct CumulativeSum2D {
    vector< vector< T > > data;

    CumulativeSum2D() {}
    CumulativeSum2D(int W, int H) : data(W + 1, vector< T >(H + 1, 0)) {}

    void add(int x, int y, T z) {
        ++x, ++y;
        if (x >= data.size() || y >= data[0].size()) return;
        data[x][y] += z;
    }

    void build() {
        for (int i = 1; i < data.size(); i++) {
            for (int j = 1; j < data[i].size(); j++) {
                data[i][j] += data[i][j - 1] + data[i - 1][j] - data[i - 1][j - 1];
            }
        }
    }

    T query(int sx, int sy, int gx, int gy) const {
        return (data[gx][gy] - data[sx][gy] - data[gx][sy] + data[sx][sy]);
    }
};

#ifdef HAVE_OPENCV_HIGHGUI
void vis(
    int N, int C,
    const Grid<char>& grid,
    const vector<pii>& edges, const vector<pii>& pts_src, const vector<pii>& pts_dst,
    int delay = 0
) {

    string color_str[6] = { "FFFFFF", "CC0A0A", "3A0BD6", "00BFB6", "73D60B", "CCBA0C" };
    cv::Scalar colors[6];
    for (int i = 0; i < 6; i++) {
        int x;
        sscanf(color_str[i].c_str(), "%x", &x);
        int r = x >> 16, g = x >> 8 & 0xFF, b = x & 0xFF;
        colors[i] = cv::Scalar(b, g, r);
    }

    auto alpha = [](const cv::Scalar& c, double a) {
        cv::Scalar w(255, 255, 255);
        cv::Scalar res;
        for (int i = 0; i < 3; i++) res[i] = std::clamp(round(a * c[i] + (1 - a) * w[i]), 0.0, 255.0);
        return res;
    };

    int gsz = 30;
    cv::Mat_<cv::Vec3b> img(gsz * N, gsz * N, cv::Vec3b(255, 255, 255));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (!grid[i + 1][j + 1]) continue;
            int c = grid[i + 1][j + 1];
            cv::Rect roi(j * gsz, i * gsz, gsz, gsz);
            cv::rectangle(img, roi, alpha(colors[c], 0.3), cv::FILLED);
        }
    }

    Grid<bool> on_edge = {};
    for (auto [u, v] : edges) {
        auto [ui, uj] = pts_dst[u];
        auto [vi, vj] = pts_dst[v];
        //dump(ui, uj, vi, vj);
        int y1 = (ui - 1) * gsz + gsz / 2, x1 = (uj - 1) * gsz + gsz / 2;
        int y2 = (vi - 1) * gsz + gsz / 2, x2 = (vj - 1) * gsz + gsz / 2;
        cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), colors[C]);
        if (ui == vi || uj == vj) {
            int d = get_dir(ui, uj, vi, vj);
            on_edge[ui][uj] = true;
            while (ui != vi || uj != vj) {
                ui += di[d]; uj += dj[d];
                on_edge[ui][uj] = true;
            }
        }
    }

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            if (on_edge[i][j] && grid[i][j] && grid[i][j] != C) {
                cv::Rect roi((j - 1) * gsz, (i - 1) * gsz, gsz, gsz);
                cv::rectangle(img, roi, alpha(colors[C], 1.0), 2);
            }
        }
    }

    for (int u = 0; u < pts_src.size(); u++) {
        if (pts_src[u] == pts_dst[u]) continue;
        auto [ui, uj] = pts_src[u];
        auto [vi, vj] = pts_dst[u];
        int y1 = (ui - 1) * gsz + gsz / 2, x1 = (uj - 1) * gsz + gsz / 2;
        int y2 = (vi - 1) * gsz + gsz / 2, x2 = (vj - 1) * gsz + gsz / 2;
        cv::arrowedLine(img, cv::Point(x1, y1), cv::Point(x2, y2), colors[C], 2, 8, 0, 0.2);
    }

    cv::imshow("img", img);
    cv::waitKey(delay);

}
#endif

struct TreeBuilder {

    struct Result {
        bool succeed;
        int C;
        vector<pii> edges;
        vector<pii> pts_src;
        vector<pii> pts_dst;
    };

    InputPtr input;
    int N, C, V, E;
    Grid<char> grid;
    CumulativeSum2D<int> cumu;
    vector<vector<int>> G;

    vector<pii> edges;
    vector<pii> pts_src;
    vector<pii> pts_dst;

    TreeBuilder(InputPtr input, int C) : input(input), C(C) {}

    Result run() {
        init();
        create_mst();
        bool succeed = align();
        return { succeed, C, edges, pts_src, pts_dst };
    }

    void init() {

        N = input->N;
        V = 0;
        grid = input->grid;
        cumu = CumulativeSum2D<int>(N + 2, N + 2);
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                if (!grid[i][j]) continue;
                if (grid[i][j] == C) {
                    pts_src.emplace_back(i, j);
                    V++;
                }
                cumu.add(i, j, 1);
            }
        }
        cumu.build();
        E = V - 1;

    }

    inline int calc_mst_cost(int u, int v) const {
        auto [ui, uj] = pts_src[u];
        auto [vi, vj] = pts_src[v];
        int cost1 = std::min(abs(ui - vi), abs(uj - vj));
        int cost2 = cumu.query(std::min(ui, vi), std::min(uj, vj), std::max(ui, vi) + 1, std::max(uj, vj) + 1) - 2;
        return cost1 + cost2;
    }

    bool cross_check(int u1, int v1) {
        // 今まで追加した辺と交差するか？
        int y1, x1, y2, x2, dy1, dx1;
        std::tie(y1, x1) = pts_src[u1];
        std::tie(y2, x2) = pts_src[v1];
        dy1 = y2 - y1;
        dx1 = x2 - x1;
        auto f = [&](int x, int y) { return dy1 * x - dx1 * y + y1 * dx1 - x1 * dy1; };
        for (auto [u2, v2] : edges) {
            int y3, x3, y4, x4;
            std::tie(y3, x3) = pts_src[u2];
            std::tie(y4, x4) = pts_src[v2];
            int z1 = f(x3, y3), z2 = f(x4, y4);
            if (z1 * z2 >= 0) continue;
            int dy2 = y4 - y3, dx2 = x4 - x3;
            auto g = [&](int x, int y) { return dy2 * x - dx2 * y + y3 * dx2 - x3 * dy2; };
            z1 = g(x1, y1); z2 = g(x2, y2);
            if (z1 * z2 < 0) return true;
        }
        return false;
    }

    void create_mst() {

        vector<std::tuple<int, int, int>> cands;
        for (int u = 0; u < V - 1; u++) {
            for (int v = u + 1; v < V; v++) {
                cands.emplace_back(calc_mst_cost(u, v), u, v);
            }
        }
        std::sort(cands.begin(), cands.end());

        UnionFind tree(V);
        G.resize(V);
        for (auto [c, u, v] : cands) {
            if (!tree.same(u, v) && !cross_check(u, v)) {
                edges.emplace_back(u, v);
                tree.unite(u, v);
                G[u].push_back(v);
                G[v].push_back(u);
            }
        }

    }

    inline int calc_align_cost(int i1, int j1, int i2, int j2) const {
        return std::min(abs(i1 - i2), abs(j1 - j2));
    };

    inline int calc_move_cost(int i1, int j1, int i2, int j2) const {
        return abs(i1 - i2) + abs(j1 - j2);
    };

    bool align() {

        pts_dst = pts_src;
        bool used[NN][NN] = {};
        for (auto [i, j] : pts_dst) used[i][j] = true; // 同じ点を目的地としないようにする

        constexpr int coeff = 3;

        int cost = 0;
        for (auto [u, v] : edges) {
            auto [ui, uj] = pts_dst[u];
            auto [vi, vj] = pts_dst[v];
            cost += calc_align_cost(ui, uj, vi, vj) * coeff;
        }

        auto get_temp = [](double startTemp, double endTemp, double t, double T) {
            return endTemp + (startTemp - endTemp) * (T - t) / T;
        };

        int num_loop = 1000000;
        vector<int> dirs({ 0, 1, 2, 3 });
        for (int loop = 0; loop < num_loop; loop++) {

            //if (!(loop & 0xFFFF)) dump(loop, cost);

            int u = rnd.next_int(V), d = -1;
            auto [ui, uj] = pts_dst[u];
            shuffle_vector(dirs, rnd);
            for (int d_ : dirs) {
                int ni = ui + di[d_], nj = uj + dj[d_];
                if (0 < ni && ni <= N && 0 < nj && nj <= N && !used[ni][nj]) {
                    d = d_;
                    break;
                }
            }
            if (d == -1) continue;

            int diff = 0;

            for (int v : G[u]) {
                auto [vi, vj] = pts_dst[v];
                diff -= calc_align_cost(ui, uj, vi, vj) * coeff;
            }
            diff -= calc_move_cost(pts_src[u].first, pts_src[u].second, ui, uj);

            ui += di[d]; uj += dj[d];

            for (int v : G[u]) {
                auto [vi, vj] = pts_dst[v];
                diff += calc_align_cost(ui, uj, vi, vj) * coeff;
            }
            diff += calc_move_cost(pts_src[u].first, pts_src[u].second, ui, uj);

            double temp = get_temp(2.0, 0.0, loop, num_loop);
            double prob = exp(-diff / temp);

            if (rnd.next_double() < prob) {
                used[pts_dst[u].first][pts_dst[u].second] = false;
                pts_dst[u] = { ui, uj };
                used[pts_dst[u].first][pts_dst[u].second] = true;
                cost += diff;
            }
        }
        //dump(cost);

        for (auto [u, v] : edges) {
            auto [ui, uj] = pts_dst[u];
            auto [vi, vj] = pts_dst[v];
            if (ui != vi && uj != vj) return false;
        }
        return true;
    }

};

struct ClusterBuilder {

    InputPtr input;

    int N, C;
    Grid<char> grid;
    vector<pii> edges;
    vector<pii> pts_src;
    vector<pii> pts_dst;
    Grid<int> smap;

    Grid<bool> on_edge;

    vector<MoveAction> moves;

    ClusterBuilder(InputPtr input, const TreeBuilder::Result& tree_res) :
        input(input), N(input->N), C(tree_res.C), grid(input->grid),
        edges(tree_res.edges), pts_src(tree_res.pts_src), pts_dst(tree_res.pts_dst), smap(), on_edge()
    {
        memset(smap.data(), -1, sizeof(int) * NN * NN);
        for (int id = 0; id < pts_src.size(); id++) {
            auto [i, j] = pts_src[id];
            smap[i][j] = id;
        }
        for (auto [u, v] : edges) {
            auto [ui, uj] = pts_dst[u];
            auto [vi, vj] = pts_dst[v];
            int d = get_dir(ui, uj, vi, vj);
            on_edge[ui][uj] = true;
            while (ui != vi || uj != vj) {
                ui += di[d]; uj += dj[d];
                on_edge[ui][uj] = true;
            }
        }
    }

    void move(int i1, int j1, int i2, int j2) {
        moves.emplace_back(i1, j1, i2, j2);
        std::swap(grid[i1][j1], grid[i2][j2]);
        std::swap(smap[i1][j1], smap[i2][j2]);
        if (smap[i1][j1] != -1) pts_src[smap[i1][j1]] = { i1, j1 };
        if (smap[i2][j2] != -1) pts_src[smap[i2][j2]] = { i2, j2 };
    }

    vector<ConnectAction> connect() {

        bool nmap[NN][NN] = {};
        int tmap[NN][NN] = {};
        UnionFind tree(pts_dst.size());
        for (int id = 0; id < pts_dst.size(); id++) {
            auto [i, j] = pts_dst[id];
            nmap[i][j] = true;
            tmap[i][j] = id;
        }

        bool emap[NN][NN][4] = {};
        for (auto [u, v] : edges) {
            auto [ui, uj] = pts_dst[u];
            auto [vi, vj] = pts_dst[v];
            int d = get_dir(ui, uj, vi, vj);
            while (ui != vi || uj != vj) {
                emap[ui][uj][d] = emap[ui + di[d]][uj + dj[d]][(d + 2) & 3] = true;
                ui += di[d]; uj += dj[d];
            }
        }

        auto [si, sj] = pts_dst.front();
        std::queue<std::tuple<int, int, int>> qu;
        bool used[NN][NN][4] = {};
        for (int d = 0; d < 4; d++) {
            if (emap[si][sj][d]) {
                qu.emplace(si, sj, d);
                used[si][sj][d] = true;
            }
        }

        vector<ConnectAction> connects;
        while (!qu.empty()) {
            auto [i, j, d] = qu.front(); qu.pop();
            int ni = i, nj = j;
            do {
                used[ni][nj][d] = used[ni + di[d]][nj + dj[d]][(d + 2) & 3] = true;
                ni += di[d]; nj += dj[d];
            } while (!nmap[ni][nj]);
            {
                int u = tmap[i][j], v = tmap[ni][nj];
                if (!tree.same(u, v)) {
                    tree.unite(u, v);
                    connects.emplace_back(i, j, ni, nj);
                }
            }
            for (int nd = 0; nd < 4; nd++) {
                if (used[ni][nj][nd] || !emap[ni][nj][nd]) continue;
                qu.emplace(ni, nj, nd);
                used[ni][nj][nd] = true;
            }
        }

        return connects;
    }

    Result run() {
        bool ok = false;
        //vis(N, C, grid, edges, pts_src, pts_dst);
        for (int trial = 0; trial < 3; trial++) {
            bool update;
            while (true) {
                update = false;
                update |= soft_move();
                update |= soft_remove();
                if (!update) break;
            }
            hard_move();
            hard_remove();
            ok = check();
            if (ok) break;
        }
        if (!ok) return {};
        //vis(N, C, grid, edges, pts_src, pts_dst);
        auto connects = connect();
        if (moves.size() + connects.size() > input->K * 100) return {};
        return { moves, connects };
    }

    bool check() const {
        bool ok = true;
        if (pts_src != pts_dst) return false;
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                if (on_edge[i][j] && grid[i][j] && grid[i][j] != C) {
                    ok = false;
                }
            }
        }
        return ok;
    }

    bool hard_remove(int si, int sj) {
        // (si, sj) 以外の on_edge を踏まずに空マスに到達するような最短パスに沿って移動
        bool seen[NN][NN] = {};
        int prev[NN][NN]; Fill(prev, -1);
        std::queue<pii> qu({ { si, sj } });
        seen[si][sj] = true;
        int ti = -1, tj = -1;
        while (!qu.empty() && ti == -1) {
            auto [i, j] = qu.front(); qu.pop();
            for (int d = 0; d < 4; d++) {
                int ni = i + di[d], nj = j + dj[d];
                if (grid[ni][nj] == -1 || seen[ni][nj] || on_edge[ni][nj]) continue;
                qu.emplace(ni, nj);
                seen[ni][nj] = true;
                prev[ni][nj] = d;
                if (!grid[ni][nj]) {
                    ti = ni; tj = nj;
                    break;
                }
            }
        }
        if (ti == -1) return false;
        vector<pii> path; // TODO: いらない
        if (ti != -1) {
            int i = ti, j = tj, d = prev[ti][tj];
            path.emplace_back(i, j);
            while (d != -1) {
                i -= di[d]; j -= dj[d]; d = prev[i][j];
                path.emplace_back(i, j);
            }
            for (int i = 0; i + 1 < path.size(); i++) {
                auto [i2, j2] = path[i];
                auto [i1, j1] = path[i + 1];
                if (!grid[i1][j1]) continue;
                move(i1, j1, i2, j2);
            }
        }
        return true;
    }

    void hard_remove() {
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                if (!(grid[i][j] && grid[i][j] != C && on_edge[i][j])) continue;
                hard_remove(i, j);
            }
        }
    }

    bool hard_move(int id) {
        auto [si, sj] = pts_src[id];
        auto [ti, tj] = pts_dst[id];
        // なるべく空マスを通るよう適当に
        vector<pii> path({ {si, sj} });
        while (si != ti || sj != tj) {
            int mindist = INT_MAX, mind = -1;
            for (int d = 0; d < 4; d++) {
                int ni = si + di[d], nj = sj + dj[d];
                int dist = 2 * (abs(ni - ti) + abs(nj - tj)) + int(grid[ni][nj] != 0);
                if (chmin(mindist, dist)) {
                    mind = d;
                }
            }
            si += di[mind]; sj += dj[mind];
            path.emplace_back(si, sj);
        }
        for (int pid = 0; pid + 1 < path.size(); pid++) {
            auto [i1, j1] = path[pid];
            auto [i2, j2] = path[pid + 1];
            if (!grid[i2][j2]) {
                // 空マスにはそのまま移動
                move(i1, j1, i2, j2);
                continue;
            }
            // (i2, j2) に存在するブロックを除外する
            bool pb = on_edge[i1][j1];
            on_edge[i1][j1] = true; // 一時的に fix
            bool ok = hard_remove(i2, j2);
            on_edge[i1][j1] = pb;
            if (!ok) return false;
            move(i1, j1, i2, j2);
        }
        return true;
    }

    void hard_move() {
        for (int id = 0; id < pts_src.size(); id++) {
            if (pts_src[id] == pts_dst[id]) continue;
            hard_move(id);
        }
    }

    bool soft_remove() {

        bool update = false;

        for (int si = 1; si <= N; si++) {
            for (int sj = 1; sj <= N; sj++) {
                if (!(grid[si][sj] && grid[si][sj] != C && on_edge[si][sj])) continue;
                // edge 上にある余計なブロックを移動
                bool seen[NN][NN] = {};
                int prev[NN][NN]; Fill(prev, -1);
                std::queue<pii> qu({ {si, sj} });
                seen[si][sj] = true;
                int ti = -1, tj = -1;
                while (!qu.empty() && ti == -1) {
                    auto [i, j] = qu.front(); qu.pop();
                    for (int d = 0; d < 4; d++) {
                        int ni = i + di[d], nj = j + dj[d];
                        if (seen[ni][nj] || grid[ni][nj]) continue;
                        qu.emplace(ni, nj);
                        seen[ni][nj] = true;
                        prev[ni][nj] = d;
                        if (!on_edge[ni][nj]) {
                            ti = ni; tj = nj;
                            break;
                        }
                    }
                }
                vector<pii> path;
                if (ti != -1) {
                    update = true;
                    int i = ti, j = tj, d = prev[ti][tj];
                    path.emplace_back(i, j);
                    while (d != -1) {
                        i -= di[d]; j -= dj[d]; d = prev[i][j];
                        path.emplace_back(i, j);
                    }
                    reverse(path.begin(), path.end());
                    for (int i = 0; i + 1 < path.size(); i++) {
                        auto [i1, j1] = path[i];
                        auto [i2, j2] = path[i + 1];
                        move(i1, j1, i2, j2);
                    }
                }
            }
        }

        return update;
    }

    bool soft_move() {

        bool update = false;

        for (int id = 0; id < pts_src.size(); id++) {
            if (pts_src[id] == pts_dst[id]) continue;
            auto [si, sj] = pts_src[id];
            auto [ti, tj] = pts_dst[id];
            // (si, sj) から (ti, tj) まで移動可能なら移動させる
            bool seen[NN][NN] = {};
            int prev[NN][NN]; Fill(prev, -1);
            std::queue<pii> qu({ pts_src[id] });
            seen[si][sj] = true;
            bool complete = false;
            while (!qu.empty() && !complete) {
                auto [i, j] = qu.front(); qu.pop();
                for (int d = 0; d < 4; d++) {
                    int ni = i + di[d], nj = j + dj[d];
                    if (seen[ni][nj] || grid[ni][nj]) continue;
                    seen[ni][nj] = true;
                    prev[ni][nj] = d;
                    if (ni == ti && nj == tj) {
                        complete = true;
                        break;
                    }
                    qu.emplace(ni, nj);
                }
            }
            vector<pii> path;
            if (complete) {
                update = true;
                int i = ti, j = tj, d = prev[ti][tj];
                path.emplace_back(i, j);
                while (d != -1) {
                    i -= di[d]; j -= dj[d]; d = prev[i][j];
                    path.emplace_back(i, j);
                }
                reverse(path.begin(), path.end());
                for (int i = 0; i + 1 < path.size(); i++) {
                    auto [i1, j1] = path[i];
                    auto [i2, j2] = path[i + 1];
                    move(i1, j1, i2, j2);
                }
            }
        }

        return update;
    }

};


Result solve(InputPtr input) {

    int best_score = -1;
    Result best_res;
    for (int c = 1; c <= input->K; c++) {
        TreeBuilder tb(input, c);
        auto tb_res = tb.run();
        if (!tb_res.succeed) continue;
        ClusterBuilder cb(input, tb_res);
        auto cb_res = cb.run();
        if (!cb_res.move.empty()) {

            int N = input->N;
            auto grid = cb.grid;
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    if (cb.on_edge[i][j]) grid[i][j] = -1;
                }
            }
            State state(input->N, input->K, grid);
            {
                vector<ConnectAction> conn;
                int rem = input->K * 100 - cb_res.move.size() - cb_res.connect.size(), score = 0;
                while (rem) {
                    auto cs = state.get_max_cluster();
                    if (cs.size() == 1) break;
                    int nc = state.greedy_connect(cs.front(), conn, rem);
                    score += nc * (nc - 1) / 2;
                }
                for (const auto& co : conn) cb_res.connect.push_back(co);
            }

            int score = calc_score(input, cb_res);
            if (chmax(best_score, score)) {
                best_res = cb_res;
                dump(c, score);
            }
        }
    }

    Solver solver(input);
    {
        auto ret = solver.solve();
        int score = calc_score(input, ret);
        if (chmax(best_score, score)) {
            best_res = ret;
            //dump(score);
        }
    }

    //dump(best_score);

    return best_res;
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
            auto res = solve(input);
            print_answer(out, res);

            {
                mtx.lock();
                scores[seed] = calc_score(input, res);
                cerr << seed << ": " << scores[seed] << '\n';
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
    std::ifstream ifs(R"(tools\in\0010.txt)");
    std::ofstream ofs(R"(tools\out\0010.txt)");
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
    auto res = solve(input);
    print_answer(out, res);
#endif

    return 0;
}