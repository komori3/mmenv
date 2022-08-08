//#define NDEBUG
#include "bits/stdc++.h"
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#ifdef _MSC_VER
#include <ppl.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif
//#define NDEBUG
#include "bits/stdc++.h"
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#ifdef _MSC_VER
#include <ppl.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

/* type */
using uint = unsigned; using ll = long long; using ull = unsigned long long; using pii = std::pair<int, int>; using pll = std::pair<ll, ll>;
/* io */
namespace aux { // print tuple
    template<typename Ty, unsigned N, unsigned L> struct tp { static void print(std::ostream& os, const Ty& v) { os << std::get<N>(v) << ", "; tp<Ty, N + 1, L>::print(os, v); } };
    template<typename Ty, unsigned N> struct tp<Ty, N, N> { static void print(std::ostream& os, const Ty& v) { os << std::get<N>(v); } };
}
template<typename... Tys> std::ostream& operator<<(std::ostream& os, const std::tuple<Tys...>& t) { os << "["; aux::tp<std::tuple<Tys...>, 0, sizeof...(Tys) - 1>::print(os, t); os << "]"; return os; }
template <typename _KTy, typename _Ty> std::ostream& operator << (std::ostream& o, const std::pair<_KTy, _Ty>& m) { o << "[" << m.first << ", " << m.second << "]"; return o; }
template <typename _KTy, typename _Ty> std::ostream& operator << (std::ostream& o, const std::map<_KTy, _Ty>& m) { if (m.empty()) { o << "[]"; return o; } o << "[" << *m.begin(); for (auto itr = ++m.begin(); itr != m.end(); itr++) { o << ", " << *itr; } o << "]"; return o; }
template <typename _KTy, typename _Ty> std::ostream& operator << (std::ostream& o, const std::unordered_map<_KTy, _Ty>& m) { if (m.empty()) { o << "[]"; return o; } o << "[" << *m.begin(); for (auto itr = ++m.begin(); itr != m.end(); itr++) { o << ", " << *itr; } o << "]"; return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::vector<_Ty>& v) { if (v.empty()) { o << "[]"; return o; } o << "[" << v.front(); for (auto itr = ++v.begin(); itr != v.end(); itr++) { o << ", " << *itr; } o << "]"; return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::deque<_Ty>& v) { if (v.empty()) { o << "[]"; return o; } o << "[" << v.front(); for (auto itr = ++v.begin(); itr != v.end(); itr++) { o << ", " << *itr; } o << "]"; return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::set<_Ty>& s) { if (s.empty()) { o << "[]"; return o; } o << "[" << *(s.begin()); for (auto itr = ++s.begin(); itr != s.end(); itr++) { o << ", " << *itr; } o << "]"; return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::unordered_set<_Ty>& s) { if (s.empty()) { o << "[]"; return o; } o << "[" << *(s.begin()); for (auto itr = ++s.begin(); itr != s.end(); itr++) { o << ", " << *itr; }	o << "]"; return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::stack<_Ty>& s) { if (s.empty()) { o << "[]"; return o; } std::stack<_Ty> t(s); o << "[" << t.top(); t.pop(); while (!t.empty()) { o << ", " << t.top(); t.pop(); } o << "]";	return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::list<_Ty>& l) { if (l.empty()) { o << "[]"; return o; } o << "[" << l.front(); for (auto itr = ++l.begin(); itr != l.end(); ++itr) { o << ", " << *itr; } o << "]"; return o; }
template <typename _KTy, typename _Ty> std::istream& operator >> (std::istream& is, std::pair<_KTy, _Ty>& m) { is >> m.first >> m.second; return is; }
template <typename _Ty> std::istream& operator >> (std::istream& is, std::vector<_Ty>& v) { for (size_t t = 0; t < v.size(); t++) is >> v[t]; return is; }
template <typename _Ty> std::istream& operator >> (std::istream& is, std::deque<_Ty>& v) { for (size_t t = 0; t < v.size(); t++) is >> v[t]; return is; }
template<typename T, size_t N> std::ostream& operator<<(std::ostream& o, const std::array<T, N>& a) { if (a.empty()) { o << "[]"; return o; } o << "[" << a[0]; for (int i = 1; i < N; i++) o << ", " << a[i]; o << "]"; return o; }
/* fill */
template<typename A, size_t N, typename T>
void Fill(A(&array)[N], const T& val) {
    std::fill((T*)array, (T*)(array + N), val);
}
/* format */
template <typename ... Args>
std::string format(const std::string& fmt, Args ... args) {
    size_t len = std::snprintf(nullptr, 0, fmt.c_str(), args ...);
    std::vector<char> buf(len + 1);
    std::snprintf(&buf[0], len + 1, fmt.c_str(), args ...);
    return std::string(&buf[0], &buf[0] + len);
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
    double elapsedMs() { return (time() - t - paused) * 1000.0; }
} timer;
/* rand */
struct Xorshift {
    uint64_t x = 88172645463325252LL;
    Xorshift(int seed = 0) { set_seed(seed); }
    void set_seed(unsigned seed, int rep = 100) {
        x = (seed + 1) * 10007;
        for (int i = 0; i < rep; i++) next_int();
    }
    unsigned next_int() {
        x = x ^ (x << 7);
        return x = x ^ (x >> 9);
    }
    unsigned next_int(unsigned mod) {
        x = x ^ (x << 7);
        x = x ^ (x >> 9);
        return x % mod;
    }
    unsigned next_int(unsigned l, unsigned r) {
        x = x ^ (x << 7);
        x = x ^ (x >> 9);
        return x % (r - l + 1) + l;
    }
    double next_double() {
        return double(next_int()) / UINT_MAX;
    }
    double next_double(double l, double r) {
        return next_double() * (r - l) + l;
    }
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

constexpr int dr[] = { 0, -1, 0, 1 };
constexpr int dc[] = { 1, 0, -1, 0 };

constexpr char CHICKEN = 'C';
constexpr char PERSON = 'P';
constexpr char WALL = '#';
constexpr char EMPTY = '.';

struct TestCase;
using TestCasePtr = std::shared_ptr<TestCase>;
struct TestCase {
    Xorshift rnd;

    // parameter ranges
    static constexpr int MIN_N = 6, MAX_N = 30;
    static constexpr double MIN_C = 0.03, MAX_C = 0.1;
    static constexpr double MIN_P = 0.01, MAX_P = 0.03;
    static constexpr double MIN_W = 0.01, MAX_W = 0.15;
    static constexpr int MIN_PEOPLES = 4;

    // parameters
    int N;
    double ratioC, ratioP, ratioW;
    int num_chickens, num_persons;

    // grid
    std::vector<std::string> grid;

    inline bool is_inside(int r, int c) const {
        return 0 <= r && r < N && 0 <= c && c < N;
    }

    bool is_reachable() const {
        std::vector<std::vector<bool>> seen(N, std::vector<bool>(N, false));
        int sr, sc;
        [&]() {
            for (int r = 0; r < N; r++) {
                for (int c = 0; c < N; c++) {
                    if (grid[r][c] == EMPTY) {
                        sr = r; sc = c; return;
                    }
                }
            }
        }();
        std::queue<pii> qu;
        qu.emplace(sr, sc);
        seen[sr][sc] = true;
        while (!qu.empty()) {
            int r, c; std::tie(r, c) = qu.front(); qu.pop();
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d], nc = c + dc[d];
                if (!is_inside(nr, nc) || seen[nr][nc] || grid[nr][nc] == WALL) continue;
                qu.emplace(nr, nc);
                seen[nr][nc] = true;
            }
        }
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] != WALL && !seen[r][c]) {
                    return false;
                }
            }
        }
        return true;
    }

    TestCase(unsigned seed, bool verbose = false) : rnd(seed) {
        N = rnd.next_int(MIN_N, MAX_N);
        ratioC = rnd.next_double(MIN_C, MAX_C);
        ratioP = rnd.next_double(MIN_P, MAX_P);
        ratioW = rnd.next_double(MIN_W, MAX_W);
        if (seed == 1) N = MIN_N;
        if (seed == 2) N = MAX_N;
        grid.resize(N, std::string(N, '.'));
        while (true) {
            num_chickens = 0;
            num_persons = 0;
            bool seen_empty = false;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    double a = rnd.next_double();
                    if (a < ratioC) {
                        grid[i][j] = CHICKEN;
                        num_chickens++;
                    }
                    else if (a < ratioC + ratioP) {
                        grid[i][j] = PERSON;
                        num_persons++;
                    }
                    else if (a < ratioC + ratioP + ratioW) {
                        grid[i][j] = WALL;
                    }
                    else {
                        grid[i][j] = EMPTY;
                        seen_empty = true;
                    }
                }
            }
            if (num_chickens > 0 && num_persons >= MIN_PEOPLES && seen_empty && is_reachable()) break;
        }
        if (verbose) {
            std::cerr << format("Grid Size, N = %d\n", N);
            std::cerr << format("Chicken ratio, ratioC = %f\n", ratioC);
            std::cerr << format("Person ratio, ratioP = %f\n", ratioP);
            std::cerr << format("Wall ratio, ratioW = %f\n\n", ratioW);
            std::cerr << format("Grid:\n");
            for (const auto& s : grid) std::cerr << s << '\n';
        }
    }

    TestCase(unsigned seed, const std::vector<std::string>& grid, bool verbose = false) : N(grid.size()), grid(grid) {
        rnd.set_seed(seed);
        num_chickens = 0;
        num_persons = 0;
        int num_walls = 0;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] == PERSON) num_persons++;
                else if (grid[r][c] == CHICKEN) num_chickens++;
                else if (grid[r][c] == WALL) num_walls++;
            }
        }
        ratioC = double(num_chickens) / (N * N);
        ratioP = double(num_persons) / (N * N);
        ratioW = double(num_walls) / (N * N);
    }
};

using Move = std::tuple<int, int, int, int>;

struct Tester;
using TesterPtr = std::shared_ptr<Tester>;
struct Tester {
    virtual int get_N() const = 0;
    virtual std::vector<std::string> get_grid() const = 0;
    virtual bool update(const std::vector<Move>& moves, bool verbose = false) = 0;
    virtual std::pair<int, std::vector<std::string>> load() = 0;
    virtual void flush() const = 0;
};

struct LocalTester;
using LocalTesterPtr = std::shared_ptr<LocalTester>;
struct LocalTester : Tester {

    TestCasePtr tc;

    int N;
    std::vector<std::string> grid;
    int num_chickens;
    int num_persons;

    int num_turns;
    std::vector<int> catch_turns;
    int global_score;
    std::vector<int> catches;

    std::vector<std::vector<bool>> used;
    std::vector<std::vector<bool>> bad;
    std::vector<int> ind;
    std::vector<int> dir;
    int max_turns;

    LocalTester(TestCasePtr tc) :
        tc(tc), N(tc->N), grid(tc->grid),
        num_chickens(tc->num_chickens), num_persons(tc->num_persons),
        num_turns(0), catch_turns(num_chickens, 0), global_score(0),
        used(N, std::vector<bool>(N, false)),
        bad(N, std::vector<bool>(N, false)),
        ind(N* N), dir(4), max_turns(N* N)
    {
        for (int i = 0; i < N * N; i++) ind[i] = i;
        for (int d = 0; d < 4; d++) dir[d] = d;
    }

    int get_N() const { return N; }

    std::vector<std::string> get_grid() const { return grid; }

    inline bool is_inside(int r, int c) const {
        return 0 <= r && r < N && 0 <= c && c < N;
    }

    inline bool are_neighbors(int r1, int c1, int r2, int c2) const {
        return abs(r1 - r2) + abs(c1 - c2) == 1;
    }

    bool update(const std::vector<Move>& moves, bool verbose = false) {
        catches.clear();
        if (verbose) std::cerr << format("Turn %d:\n", num_turns);
        if (num_turns > max_turns) {
            std::cerr << format("Used more than %d turns\n", max_turns);
            return false;
        }
        num_turns++;
        int num_moves = moves.size();
        for (int i = 0; i < N; i++) for (int k = 0; k < N; k++) used[i][k] = false;
        for (int i = 0; i < moves.size(); i++) {
            int r1, c1, r2, c2; std::tie(r1, c1, r2, c2) = moves[i];
            if (!is_inside(r1, c1) || !is_inside(r2, c2) || !are_neighbors(r1, c1, r2, c2)) {
                std::cerr << format("Invalid coordinates of move: %d %d %d %d\n", r1, c1, r2, c2);
                return false;
            }
            if (used[r1][c1]) {
                std::cerr << format("This person has already moved this turn: %d %d %d %d\n", r1, c1, r2, c2);
                return false;
            }
            if (grid[r1][c1] != PERSON) {
                std::cerr << format("You can only move people: %d %d %d %d\n", r1, c1, r2, c2);
                return false;
            }
            if (!(grid[r2][c2] == EMPTY || grid[r2][c2] == CHICKEN)) {
                std::cerr << format("You can only move into empty or chicken cells: %d %d %d %d\n", r1, c1, r2, c2);
                return false;
            }

            if (grid[r2][c2] == CHICKEN) {
                num_chickens--;
                catch_turns[num_chickens] = num_turns;
                global_score += N * N - (num_turns - 1);
                catches.push_back(r2 * N + c2);
            }
            grid[r1][c1] = EMPTY;
            grid[r2][c2] = PERSON;
            used[r2][c2] = true;
            if (verbose) std::cerr << format("\tMove %d: %d %d %d %d\n", i + 1, r1, c1, r2, c2);
        }
        for (int i = 0; i < N; i++) for (int k = 0; k < N; k++) used[i][k] = false;
        for (int i = 0; i < N; i++) for (int k = 0; k < N; k++) bad[i][k] = false;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] == PERSON) {
                    for (int d = 0; d < 4; d++) {
                        int nr = r + dr[d], nc = c + dc[d];
                        if (is_inside(nr, nc)) bad[nr][nc] = true;
                    }
                }
            }
        }
        shuffle_vector(ind, tc->rnd);
        for (int i = 0; i < ind.size(); i++) {
            int r = ind[i] / N, c = ind[i] % N;
            if (grid[r][c] == CHICKEN && !used[r][c]) {
                shuffle_vector(dir, tc->rnd);
                for (int d = 0; d < 4; d++) {
                    int r2 = r + dr[dir[d]], c2 = c + dc[dir[d]];
                    if (is_inside(r2, c2) && grid[r2][c2] == EMPTY && !bad[r2][c2]) {
                        grid[r][c] = EMPTY;
                        grid[r2][c2] = CHICKEN;
                        used[r2][c2] = true;
                        break;
                    }
                }
            }
        }
        return true;
    }

    std::pair<int, std::vector<std::string>> load() {
        return { (int)timer.elapsedMs(), grid };
    }

    void flush() const {
        std::cerr << format("Score = %.1f\n", (double)global_score);
    }
};

struct IOTester;
using IOTesterPtr = std::shared_ptr<IOTester>;
struct IOTester : Tester {
    std::istream& in;
    std::ostream& out;
    int N;
    std::vector<std::string> grid;
    IOTester(std::istream& in, std::ostream& out) : in(in), out(out) {
        in >> N;
        grid.resize(N, std::string(N, '$'));
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                in >> grid[r][c];
            }
        }
    }
    int get_N() const { return N; }
    std::vector<std::string> get_grid() const { return grid; }
    bool update(const std::vector<Move>& moves, bool verbose = false) {
        out << moves.size() << '\n';
        for (const auto& move : moves) {
            int r1, c1, r2, c2; std::tie(r1, c1, r2, c2) = move;
            out << r1 << ' ' << c1 << ' ' << r2 << ' ' << c2 << '\n';
        }
        return true;
    }
    std::pair<int, std::vector<std::string>> load() {
        int elapsed_ms;
        in >> elapsed_ms;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                in >> grid[r][c];
            }
        }
        return { elapsed_ms, grid };
    }
    void flush() const {
        out << -1;
    }
};

struct ASSP {
    int V;
    std::vector<std::vector<int>> dist;
    std::vector<std::vector<int>> next;
    std::vector<std::vector<std::vector<int>>> nexts;
    ASSP() {}
    ASSP(const std::vector<std::vector<int>>& distance_matrix) :
        V(distance_matrix.size()), dist(distance_matrix),
        next(V, std::vector<int>(V)), nexts(V, std::vector<std::vector<int>>(V))
    {
        build();
    }
    void build() {
        for (int u = 0; u < V; u++) for (int v = 0; v < V; v++) next[u][v] = v;
        for (int k = 0; k < V; k++) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                        next[i][j] = next[i][k];
                        nexts[i][j].clear();
                        nexts[i][j].push_back(next[i][k]);
                    }
                    else if (dist[i][k] + dist[k][j] == dist[i][j]) {
                        nexts[i][j].push_back(next[i][k]);
                    }
                }
            }
        }
    }
    std::vector<int> get_path(int u, int v) const {
        std::vector<int> ret;
        for (int cur = u; cur != v; cur = next[cur][v]) ret.push_back(cur);
        ret.push_back(v);
        return ret;
    }
};

struct State;
using StatePtr = std::shared_ptr<State>;
struct State {

    int turn;
    int elapsed_ms;
    int N;
    int score;
    int max_score;

    int num_persons;
    int num_chickens;
    std::vector<std::string> grid;

    ASSP assp;

    State(TesterPtr tester) : turn(0), elapsed_ms(0), N(tester->get_N()), score(0), grid(tester->get_grid()) {
        num_persons = num_chickens = 0;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] == PERSON) {
                    num_persons++;
                }
                if (grid[r][c] == CHICKEN) {
                    num_chickens++;
                }
            }
        }
        max_score = num_chickens * N * N;
        calc_assp();
    }

    void calc_assp() {
        int V = N * N;
        auto g(grid);
        for (auto& s : g) for (auto& c : s) if (c != '.' && c != '#') c = '.';
        int inf = INT_MAX / 8;
        std::vector<std::vector<int>> d(V, std::vector<int>(V, inf));
        // diag
        for (int u = 0; u < V; u++) d[u][u] = 0;
        // yoko
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N - 1; c++) {
                if (g[r][c] != '#' && g[r][c + 1] != '#') {
                    int u = r * N + c, v = r * N + c + 1;
                    d[u][v] = d[v][u] = 1;
                }
            }
        }
        // tate
        for (int r = 0; r < N - 1; r++) {
            for (int c = 0; c < N; c++) {
                if (g[r][c] != '#' && g[r + 1][c] != '#') {
                    int u = r * N + c, v = (r + 1) * N + c;
                    d[u][v] = d[v][u] = 1;
                }
            }
        }
        assp = ASSP(d);
    }

    pii calc_next_cell(int r1, int c1, int r2, int c2) const {
        // (r1, c1) -> (r2, c2) の最短路における (r1, c1) の次のセル
        int u = r1 * N + c1, v = r2 * N + c2, n = assp.next[u][v];
        return { n / N, n % N };
    }

    std::vector<pii> calc_next_cells(int r1, int c1, int r2, int c2) const {
        std::vector<pii> ret;
        std::vector<std::tuple<int, int, int>> tup;
        for (int d = 0; d < 4; d++) {
            int nr = r1 + dr[d], nc = c1 + dc[d];
            if (!is_inside(nr, nc) || grid[nr][nc] == '#' || grid[nr][nc] == 'P') continue;
            tup.emplace_back(assp.dist[nr * N + nc][r2 * N + c2], nr, nc);
        }
        if (tup.empty()) return ret;
        sort(tup.begin(), tup.end());
        for (auto& t : tup) {
            ret.emplace_back(std::get<1>(t), std::get<2>(t));
        }
        return ret;
    }

    std::vector<pii> calc_next_cells_2(int r1, int c1, int r2, int c2) const {
        std::vector<pii> ret;
        int u = r1 * N + c1, v = r2 * N + c2;
        auto ns = assp.nexts[u][v];
        for (int rc : ns) {
            ret.emplace_back(rc / N, rc % N);
        }
        return ret;
    }

    inline bool is_inside(int r, int c) const {
        return 0 <= r && r < N && 0 <= c && c < N;
    }

    std::vector<Move> do_random_moves() {
        // ガチャガチャ移動
        // 近くに鶏がいれば捕獲
        turn++;
        std::vector<Move> moves;
        std::vector<std::vector<bool>> used(N, std::vector<bool>(N, false));
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] == 'P' && !used[r][c]) {
                    int chicken_dir = -1;
                    std::vector<int> dirs;
                    for (int d = 0; d < 4; d++) {
                        int nr = r + dr[d], nc = c + dc[d];
                        if (nr < 0 || nr >= N || nc < 0 || nc >= N || used[nr][nc] || grid[nr][nc] == '#' || grid[nr][nc] == 'P') continue;
                        if (chicken_dir == -1 && grid[nr][nc] == 'C') chicken_dir = d;
                        dirs.push_back(d);
                    }
                    if (chicken_dir != -1) {
                        // move to chicken
                        int nr = r + dr[chicken_dir], nc = c + dc[chicken_dir];
                        moves.emplace_back(r, c, nr, nc);
                        grid[r][c] = '.';
                        grid[nr][nc] = 'P';
                        used[nr][nc] = true;
                        score += N * N - (turn - 1);
                    }
                    else if (!dirs.empty()) {
                        int d = dirs[rnd.next_int(dirs.size())];
                        int nr = r + dr[d], nc = c + dc[d];
                        moves.emplace_back(r, c, nr, nc);
                        grid[r][c] = '.';
                        grid[nr][nc] = 'P';
                        used[nr][nc] = true;
                    }
                }
            }
        }
        return moves;
    }
    std::vector<Move> move_to_nearest_chicken() {
        // 最近鶏に向かう
        turn++;
        std::vector<pii> persons;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] == PERSON) {
                    persons.emplace_back(r, c);
                }
            }
        }

        std::vector<Move> moves;
        std::vector<std::vector<bool>> used(N, std::vector<bool>(N, false));

        for (const auto& e : persons) {
            int pr, pc; std::tie(pr, pc) = e;
            int mindist = INT_MAX, tr = -1, tc = -1;
            for (int r = 0; r < N; r++) {
                for (int c = 0; c < N; c++) {
                    if (grid[r][c] != CHICKEN) continue;
                    int dist = abs(pr - r) + abs(pc - c);
                    if (dist < mindist) {
                        mindist = dist;
                        tr = r;
                        tc = c;
                    }
                }
            }
            if (tr == -1) continue;
            std::vector<std::tuple<int, int, int>> tup;
            for (int d = 0; d < 4; d++) {
                int nr = pr + dr[d], nc = pc + dc[d];
                if (!is_inside(nr, nc) || used[nr][nc] || grid[nr][nc] == '#' || grid[nr][nc] == 'P') continue;
                tup.emplace_back(abs(nr - tr) + abs(nc - tc), nr, nc);
            }
            if (tup.empty()) continue;
            std::sort(tup.begin(), tup.end());
            int _, r2, c2; std::tie(_, r2, c2) = tup.front();
            grid[pr][pc] = '.';
            if (grid[r2][c2] == CHICKEN) score += N * N - (turn - 1);
            grid[r2][c2] = 'P';
            used[r2][c2] = true;
            moves.emplace_back(pr, pc, r2, c2);
        }
        return moves;
    }

    std::vector<Move> move_to_nearest_chicken_assp() {
        turn++;
        std::vector<pii> persons;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] == PERSON) {
                    persons.emplace_back(r, c);
                }
            }
        }

        std::vector<Move> moves;
        std::vector<std::vector<bool>> used(N, std::vector<bool>(N, false));

        for (const auto& e : persons) {
            int pr, pc; std::tie(pr, pc) = e;
            int mindist = INT_MAX, tr = -1, tc = -1;
            for (int r = 0; r < N; r++) {
                for (int c = 0; c < N; c++) {
                    if (grid[r][c] != CHICKEN) continue;
                    int dist = abs(pr - r) + abs(pc - c);
                    if (dist < mindist) {
                        mindist = dist;
                        tr = r;
                        tc = c;
                    }
                }
            }
            if (tr == -1) continue;

            auto cands = calc_next_cells(pr, pc, tr, tc);
            if (cands.empty()) continue;
            int r2, c2; std::tie(r2, c2) = cands.front();
            assert(grid[r2][c2] != 'P');
            grid[pr][pc] = '.';
            if (grid[r2][c2] == CHICKEN) score += N * N - (turn - 1);
            grid[r2][c2] = 'P';
            used[r2][c2] = true;
            moves.emplace_back(pr, pc, r2, c2);
        }
        return moves;
    }

    std::vector<Move> move_to_nearest_chicken_assp_2() {
        // 最近鶏に向かう
        turn++;
        std::vector<pii> persons;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] == PERSON) {
                    persons.emplace_back(r, c);
                }
            }
        }

        std::vector<Move> moves;
        std::vector<std::vector<bool>> used(N, std::vector<bool>(N, false));

        for (const auto& e : persons) {
            int pr, pc; std::tie(pr, pc) = e;
            int mindist = INT_MAX, tr = -1, tc = -1;
            for (int r = 0; r < N; r++) {
                for (int c = 0; c < N; c++) {
                    if (grid[r][c] != CHICKEN) continue;
                    int dist = abs(pr - r) + abs(pc - c);
                    if (dist < mindist) {
                        mindist = dist;
                        tr = r;
                        tc = c;
                    }
                }
            }
            if (tr == -1) continue;
            auto cands = calc_next_cells_2(pr, pc, tr, tc);
            assert(cands.size() >= 1);
            //shuffle_vector(cands, rnd);
            int r2 = -1, c2 = -1;
            for (auto rc : cands) {
                if (grid[rc.first][rc.second] == 'P') continue;
                r2 = rc.first; c2 = rc.second;
                break;
            }
            if (r2 == -1) continue;
            assert(grid[r2][c2] != 'P');
            grid[pr][pc] = '.';
            if (grid[r2][c2] == CHICKEN) score += N * N - (turn - 1);
            grid[r2][c2] = 'P';
            used[r2][c2] = true;
            moves.emplace_back(pr, pc, r2, c2);
        }
        return moves;
    }
};


struct State2;
using State2Ptr = std::shared_ptr<State2>;
struct State2 {

    int turn;
    int elapsed_ms;
    int N;
    int score;
    int max_score;

    int num_persons;
    int num_chickens;
    std::vector<std::string> grid;

    std::vector<pii> group;

    ASSP assp;

    State2(TesterPtr tester) : turn(0), elapsed_ms(0), N(tester->get_N()), score(0), grid(tester->get_grid()) {
        num_persons = num_chickens = 0;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] == PERSON) {
                    num_persons++;
                    if (group.size() < 4) group.emplace_back(r, c);
                }
                if (grid[r][c] == CHICKEN) {
                    num_chickens++;
                }
            }
        }
        max_score = num_chickens * N * N;
        calc_assp();
    }

    void calc_assp() {
        int V = N * N;
        auto g(grid);
        for (auto& s : g) for (auto& c : s) if (c != '.' && c != '#') c = '.';
        int inf = INT_MAX / 8;
        std::vector<std::vector<int>> d(V, std::vector<int>(V, inf));
        // diag
        for (int u = 0; u < V; u++) d[u][u] = 0;
        // yoko
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N - 1; c++) {
                if (g[r][c] != '#' && g[r][c + 1] != '#') {
                    int u = r * N + c, v = r * N + c + 1;
                    d[u][v] = d[v][u] = 1;
                }
            }
        }
        // tate
        for (int r = 0; r < N - 1; r++) {
            for (int c = 0; c < N; c++) {
                if (g[r][c] != '#' && g[r + 1][c] != '#') {
                    int u = r * N + c, v = (r + 1) * N + c;
                    d[u][v] = d[v][u] = 1;
                }
            }
        }
        assp = ASSP(d);
    }

    pii calc_next_cell(int r1, int c1, int r2, int c2) const {
        // (r1, c1) -> (r2, c2) の最短路における (r1, c1) の次のセル
        int u = r1 * N + c1, v = r2 * N + c2, n = assp.next[u][v];
        return { n / N, n % N };
    }

    inline bool is_inside(int r, int c) const {
        return 0 <= r && r < N && 0 <= c && c < N;
    }

    std::vector<Move> group_hunting() {
        turn++;
        pii target;
        {
            int min_dist = INT_MAX;
            for (int r = 0; r < N; r++) {
                for (int c = 0; c < N; c++) {
                    if (grid[r][c] != CHICKEN) continue;
                    int dist = 0;
                    for (auto& p : group) {
                        int pr, pc; std::tie(pr, pc) = p;
                        dist += assp.dist[pr * N + pc][r * N + c];
                    }
                    if (dist < min_dist) {
                        target = pii(r, c);
                        min_dist = dist;
                    }
                }
            }
        }

        std::vector<Move> moves;
        std::vector<std::vector<bool>> used(N, std::vector<bool>(N, false));

        for (int i = 0; i < group.size(); i++) {
            int pr, pc; std::tie(pr, pc) = group[i];
            int r2, c2;
            std::tie(r2, c2) = calc_next_cell(pr, pc, target.first, target.second);
            if (grid[r2][c2] == 'P') continue;
            grid[pr][pc] = '.';
            if (grid[r2][c2] == CHICKEN) score += N * N - (turn - 1);
            grid[r2][c2] = 'P';
            group[i].first = r2;
            group[i].second = c2;
            used[r2][c2] = true;
            moves.emplace_back(pr, pc, r2, c2);
        }
        return moves;
    }
};


void solve_(TesterPtr tester, StatePtr state) {
    for (int turn = 1; turn <= state->N * state->N; turn++) {
        auto moves = state->move_to_nearest_chicken_assp();
        bool ok = tester->update(moves);
        assert(ok);
        std::tie(state->elapsed_ms, state->grid) = tester->load();
    }
    double reletive_score = (double)state->score / state->max_score;
    dump(reletive_score);
}

void solve(TesterPtr tester, StatePtr state) {
    int num_trial = 10;
    double score_sum = 0.0;
    for (int seed = 0; seed < num_trial; seed++) {
        TestCasePtr tc = std::make_shared<TestCase>(seed, tester->get_grid(), false);
        LocalTesterPtr ltester = std::make_shared<LocalTester>(tc);
        StatePtr state2 = std::make_shared<State>(*state);
        for (int turn = 1; turn <= state2->N * state2->N; turn++) {
            auto moves = state2->move_to_nearest_chicken_assp();
            bool ok = ltester->update(moves);
            assert(ok);
            std::tie(state2->elapsed_ms, state2->grid) = ltester->load();
        }
        double relative_score = (double)state2->score / state2->max_score;
        score_sum += relative_score;
    }
    score_sum /= num_trial;

    double score_sum2 = 0.0;
    for (int seed = 0; seed < num_trial; seed++) {
        TestCasePtr tc = std::make_shared<TestCase>(seed, tester->get_grid(), false);
        LocalTesterPtr ltester = std::make_shared<LocalTester>(tc);
        StatePtr state2 = std::make_shared<State>(*state);
        for (int turn = 1; turn <= state2->N * state2->N; turn++) {
            auto moves = state2->move_to_nearest_chicken_assp_2();
            bool ok = ltester->update(moves);
            assert(ok);
            std::tie(state2->elapsed_ms, state2->grid) = ltester->load();
        }
        double relative_score = (double)state2->score / state2->max_score;
        score_sum2 += relative_score;
    }
    score_sum2 /= num_trial;

    for (int turn = 1; turn <= state->N * state->N; turn++) {
        auto moves = score_sum < score_sum2 ? state->move_to_nearest_chicken_assp() : state->move_to_nearest_chicken_assp_2();
        bool ok = tester->update(moves);
        assert(ok);
        std::tie(state->elapsed_ms, state->grid) = tester->load();
    }
}

void solve2(TesterPtr tester, State2Ptr state) {
    for (int turn = 1; turn <= state->N * state->N; turn++) {
        auto moves = state->group_hunting();
        bool ok = tester->update(moves);
        assert(ok);
        std::tie(state->elapsed_ms, state->grid) = tester->load();
    }
    double reletive_score = (double)state->score / state->max_score;
    dump(reletive_score);
}

int _main() {
    double relative_score_sum = 0.0;
    for (int seed = 1; seed <= 100; seed++) {
        TestCasePtr tc = std::make_shared<TestCase>(seed);
        LocalTesterPtr tester = std::make_shared<LocalTester>(tc);
        StatePtr state = std::make_shared<State>(tester);
        solve(tester, state);
        relative_score_sum = (double)state->score / state->max_score;
        std::cerr << format("seed = %3d, score = %8.1f, ratio = %f\n", seed, (double)tester->global_score, (double)tester->global_score / state->max_score);
    }
    dump(relative_score_sum);
    return 0;
}

int main() {

    bool local = false;
    bool verbose = false;
    TesterPtr tester;

    if (local) {
        TestCasePtr tc = std::make_shared<TestCase>(1, verbose);
        tester = std::make_shared<LocalTester>(tc);
    }
    else {
        std::istream& in = std::cin;
        std::ostream& out = std::cout;
        tester = std::make_shared<IOTester>(in, out);
    }

    StatePtr state = std::make_shared<State>(tester);
    solve(tester, state);
    //State2Ptr state = std::make_shared<State2>(tester);
    //solve2(tester, state);
    tester->flush();

    return 0;
}