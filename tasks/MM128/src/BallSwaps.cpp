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
namespace aux { // print tuple
    template<typename Ty, unsigned N, unsigned L> struct tp { static void print(std::ostream& os, const Ty& v) { os << std::get<N>(v) << ", "; tp<Ty, N + 1, L>::print(os, v); } };
    template<typename Ty, unsigned N> struct tp<Ty, N, N> { static void print(std::ostream& os, const Ty& v) { os << std::get<N>(v); } };
}
template<typename... Tys> std::ostream& operator<<(std::ostream& os, const std::tuple<Tys...>& t) { os << "["; aux::tp<std::tuple<Tys...>, 0, sizeof...(Tys) - 1>::print(os, t); os << "]"; return os; }


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
#define dump(...) do{DUMPOUT<<"  ";DUMPOUT<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPOUT<<"    ";dump_func(__VA_ARGS__);}while(0);
void dump_func() { DUMPOUT << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPOUT << head; if (sizeof...(Tail) == 0) { DUMPOUT << " "; } else { DUMPOUT << ", "; } dump_func(std::move(tail)...); }
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
/* fast queue */
class FastQueue {
    int front, back;
    int v[1 << 12];
public:
    FastQueue() : front(0), back(0) {}
    inline bool empty() { return front == back; }
    inline void push(int x) { v[front++] = x; }
    inline int pop() { return v[back++]; }
    inline void reset() { front = back = 0; }
    inline int size() { return front - back; }
} fqu;

//using namespace std;
using std::vector; using std::string; using std::cerr; using std::cout; using std::cin; using std::endl; using std::shared_ptr;

struct Point {
    int i, j;
    constexpr Point(int i = 0, int j = 0) : i(i), j(j) {}
    inline Point& operator+=(const Point& p) { i += p.i; j += p.j; return *this; }
    inline Point& operator-=(const Point& p) { i -= p.i; j -= p.j; return *this; }
    inline Point& operator-() { i = -i; j = -j; return *this; }
    inline bool operator==(const Point& p) const { return i == p.i && j == p.j; }
    inline bool operator!=(const Point& p) const { return !(*this == p); }
    inline bool operator<(const Point& p) const { return i == p.i ? j < p.j : i < p.i; }
    inline int distance(const Point& p) const { return abs(i - p.i) + abs(j - p.j); }
    std::string str() const { return "[" + std::to_string(i) + ", " + std::to_string(j) + ']'; }
    friend std::ostream& operator<<(std::ostream& o, const Point& p) { o << p.str(); return o; }
};
Point operator+(const Point& p1, const Point& p2) { return Point(p1) += p2; }
Point operator-(const Point& p1, const Point& p2) { return Point(p1) -= p2; }
int distance(const Point& p1, const Point& p2) { return p1.distance(p2); }

constexpr Point dir[] = { {0,1},{1,0},{0,-1},{-1,0} };
constexpr int di[] = { 0, 1, 0, -1 };
constexpr int dj[] = { 1, 0, -1, 0 };

using Path = vector<Point>;
using Board = vector<vector<int>>;
using Move = std::tuple<int, int, int, int>;



/* global */
int N, C;
Board g_board;
vector<int> g_count;
unsigned int g_board_mask[32];
#ifdef _MSC_VER
const vector<cv::Scalar> g_color({
    cv::Scalar(0, 0, 0),       // black (frame color)
    cv::Scalar(255, 0, 0),     // blue
    cv::Scalar(255, 0, 255),   // magenta
    cv::Scalar(128, 128, 128), // gray
    cv::Scalar(0, 0, 255),     // red
    cv::Scalar(255, 255, 0),   // cyan
    cv::Scalar(175, 175, 255), // pink
    cv::Scalar(0, 255, 0),     // green
    cv::Scalar(0, 200, 255)    // orange
    });
#endif

void init(std::istream& in) {
    in >> N >> C;
    g_board.resize(N + 2, vector<int>(N + 2, 0)); // 1-indexed
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            in >> g_board[i][j];
            g_board[i][j]++;
        }
    }
    g_count.resize(C + 1, 0); // 1-indexed
    for (int i = 0; i < N + 2; i++) {
        for (int j = 0; j < N + 2; j++) {
            g_count[g_board[i][j]]++;
        }
    }
    //for (const auto& v : g_board) cerr << v << endl;
    //cerr << g_count << endl;
    unsigned int frame_mask = (1ULL << (N + 2)) - 1;
    unsigned int inner_mask = (1ULL << (N + 1)) | 1;
    g_board_mask[0] = g_board_mask[N + 1] = frame_mask;
    for (int i = 1; i <= N; i++) g_board_mask[i] = inner_mask;
    //for (int i = 0; i < N + 2; i++) cerr << std::bitset<32>(g_board_mask[i]) << endl;
}

inline bool is_inside(const Point& p) {
    return 1 <= p.i && p.i <= N && 1 <= p.j && p.j <= N;
}

Path generate_spiral(int N) {
    Path spiral;
    vector<vector<bool>> visited(N + 2, vector<bool>(N + 2, true));
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            visited[i][j] = false;
        }
    }
    Point p(1, 1);
    int d = 0;
    spiral.push_back(p);
    visited[p.i][p.j] = true;
    while (true) {
        if (!visited[p.i + dir[d].i][p.j + dir[d].j]) {
            p += dir[d];
            spiral.push_back(p);
            visited[p.i][p.j] = true;
        }
        else if (!visited[p.i + dir[(d + 1) & 3].i][p.j + dir[(d + 1) & 3].j]) {
            d = (d + 1) & 3;
            p += dir[d];
            spiral.push_back(p);
            visited[p.i][p.j] = true;
        }
        else {
            break;
        }
    }
    return spiral;
}

Path generate_zigzag(int N) {
    Path zigzag;
    for (int i = 1; i <= N; i++) {
        if (i % 2 == 1) {
            for (int j = 1; j <= N; j++) {
                zigzag.emplace_back(i, j);
            }
        }
        else {
            for (int j = N; j >= 1; j--) {
                zigzag.emplace_back(i, j);
            }
        }
    }
    return zigzag;
}

namespace NRoute {

    struct State {
        Board S; // 初期盤面
        Board T; // 最終盤面
        Path route; // 移動する順番
        vector<vector<bool>> fixed; // 移動終了したか？

        vector<Move> moves; // 移動履歴

        State(const Board& T, const Path& route)
            : S(g_board), T(T), route(route), fixed(N + 2, vector<bool>(N + 2, false)) {
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    fixed[i][j] = false;
                }
            }
        }

        Path get_shortest_path(int x, const Point& dst) const {
            // ある数字 x をセル c に移動させる最短パス　移動禁止領域: fixed
            // dst に最も近い数字 x の場所を求めて、経路復元
            int mindist = 1e9;
            Point src;
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    if (fixed[i][j] || S[i][j] != x) continue;
                    int dist = Point(i, j).distance(dst);
                    if (dist < mindist) {
                        mindist = dist;
                        src = Point(i, j);
                    }
                }
            }

            Path path({ src });

            while (src != dst) {
                for (int d = 0; d < 4; d++) {
                    auto p = src + dir[d];
                    int dist = p.distance(dst);
                    if (!fixed[p.i][p.j] && dist < mindist) {
                        src = p;
                        path.push_back(p);
                        mindist = dist;
                        break;
                    }
                }
            }
            return path;
        }

        void do_moves(const Path& path) {
            for (int i = 0; i < (int)path.size() - 1; i++) {
                const auto& p1 = path[i];
                const auto& p2 = path[i + 1];
                std::swap(S[p1.i][p1.j], S[p2.i][p2.j]);
                moves.emplace_back(p1.i, p1.j, p2.i, p2.j);
            }
        }

        void solve() {
            // route に沿って揃えていく
            // 既に揃えられたマス以外を通って数字 T[i][j] を route[idx] に移動させるような最短パスを求める
            for (int idx = 0; idx < N * N; idx++) {
                int i = route[idx].i, j = route[idx].j;
                if (S[i][j] == T[i][j]) {
                    fixed[i][j] = true;
                    continue;
                }
                auto path = get_shortest_path(T[i][j], route[idx]);
                do_moves(path);
                fixed[i][j] = true;
            }
        }

        std::string str() const {
            std::ostringstream oss;
            oss << "--- State ---\n"
                << "N = " << N << ", C = " << C << '\n';
            for (int i = 0; i < N + 2; i++) {
                for (int j = 0; j < N + 2; j++) {
                    oss << S[i][j] << ' ';
                }
                oss << '\n';
            }
            oss << "-------------\n";
            return oss.str();
        }

        friend std::ostream& operator<<(std::ostream& o, const State& obj) {
            o << obj.str();
            return o;
        }

        void output(std::ostream& o) const {
            o << moves.size() << '\n';
            for (const auto& t : moves) {
                int i1, j1, i2, j2;
                std::tie(i1, j1, i2, j2) = t;
                o << i1 - 1 << ' ' << j1 - 1 << ' ' << i2 - 1 << ' ' << j2 - 1 << '\n';
            }
        }
    };

}

namespace NStrictTransform {
    // 初期盤面を S、最終盤面を T とすると問題は
    // 1. valid な（かつ良型の） T を求めるフェーズ
    // 2. S を T に変形するフェーズ
    // の二段階に分けられる

    // 1. はとりあえず求めるだけなら、蛇腹や螺旋の一本道に同じ色をまとめて突っ込んでいけば構築できる
    // とりあえず求めた盤面を、制約を守りながら望ましい形に変化させていくアプローチを取る
    // 盤面の良さは、暫定で S[i][j] == T[i][j] となるマスの個数とする
    // -> 後で S をベースに blur を掛けたようなマップを生成して評価関数を滑らかにするのもありかも

    struct State {
        Board T;

        State(const Board& T) : T(T) {}

        int calc_score() const {
            int score = 0;
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    score += T[i][j] == g_board[i][j];
                }
            }
            return score;
        }

        bool is_valid() const {
            // 各色 (0 除く) の連結成分数は 1
            // 色 c のセル数は g_count[c] に一致
            static unsigned int cmask;     // used color mask
            static unsigned int bmask[32]; // used cell mask
            // init
            cmask = 0;
            memcpy(bmask, g_board_mask, sizeof(unsigned int) * 32);

            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    if ((bmask[i] >> j) & 1) continue; // used
                    int c = T[i][j], cnt = 0;
                    if ((cmask >> c) & 1) return false; // connected component > 1
                    cmask |= (1U << c);
                    // bfs
                    fqu.reset();
                    fqu.push((i << 5) | j);
                    bmask[i] |= (1U << j);
                    cnt++;
                    while (!fqu.empty()) {
                        int cij = fqu.pop(), ci = (cij >> 5), cj = (cij & 0b11111);
                        for (int d = 0; d < 4; d++) {
                            int ni = ci + di[d], nj = cj + dj[d];
                            if (((bmask[ni] >> nj) & 1) || T[ni][nj] != c) continue;
                            fqu.push((ni << 5) | nj);
                            bmask[ni] |= (1U << nj);
                            cnt++;
                        }
                    }
                    if (cnt != g_count[c]) return false; // pixel count violation
                }
            }
            return true;
        }

#ifdef _MSC_VER
        void vis(int delay = 0) const {
            int grid_size = 960 / (N + 2);
            int height = grid_size * (N + 2), width = grid_size * (N + 2);
            cv::Mat_<cv::Vec3b> img(height, width, cv::Vec3b(255, 255, 255));
            for (int i = 0; i < N + 2; i++) {
                for (int j = 0; j < N + 2; j++) {
                    if (g_board[i][j] == T[i][j]) {
                        cv::rectangle(img, cv::Rect(grid_size * j, grid_size * i, grid_size, grid_size), cv::Scalar(0, 255, 255), cv::FILLED);
                    }
                    if (g_board[i][j]) {
                        cv::circle(img, cv::Point(grid_size * (j + 0.5), grid_size * (i + 0.5)), grid_size / 3, g_color[T[i][j]], cv::FILLED);
                    }
                    else {
                        cv::rectangle(img, cv::Rect(grid_size * j, grid_size * i, grid_size, grid_size), cv::Scalar(0, 0, 0), cv::FILLED);
                    }
                }
            }
            for (int i = 1; i < N + 2; i++) {
                cv::line(img, cv::Point(0, i * grid_size), cv::Point((N + 2) * grid_size, i * grid_size), cv::Scalar(0, 0, 0));
                cv::line(img, cv::Point(i * grid_size, 0), cv::Point(i * grid_size, (N + 2) * grid_size), cv::Scalar(0, 0, 0));
            }
            cv::imshow("img", img);
            cv::waitKey(delay);
        }
#endif
    };
}

namespace NFlow {

    template<typename flow_t, typename cost_t>
    struct PrimalDual {
        const cost_t INF;

        struct edge {
            int to;
            flow_t cap;
            cost_t cost;
            int rev;
            bool isrev;
            edge(int to = -1, flow_t cap = -1, cost_t cost = -1, int rev = -1, bool isrev = false)
                : to(to), cap(cap), cost(cost), rev(rev), isrev(isrev) {}
        };

        vector<vector<edge>> graph;
        vector<cost_t> potential, min_cost;
        vector<int> prevv, preve;

        PrimalDual(int V) : INF(std::numeric_limits<cost_t>::max()), graph(V) {}

        void add_edge(int from, int to, flow_t cap, cost_t cost) {
            graph[from].emplace_back(to, cap, cost, (int)graph[to].size(), false);
            graph[to].emplace_back(from, 0, -cost, (int)graph[from].size() - 1, true);
        }

        cost_t min_cost_flow(int s, int t, flow_t f) {
            int V = (int)graph.size();
            cost_t ret = 0;
            using Pi = std::pair<cost_t, int>;
            std::priority_queue<Pi, vector<Pi>, std::greater<Pi>> que;
            potential.assign(V, 0);
            preve.assign(V, -1);
            prevv.assign(V, -1);

            while (f > 0) {
                min_cost.assign(V, INF);
                que.emplace(0, s);
                min_cost[s] = 0;
                while (!que.empty()) {
                    Pi p = que.top();
                    que.pop();
                    if (min_cost[p.second] < p.first) continue;
                    for (int i = 0; i < (int)graph[p.second].size(); i++) {
                        edge& e = graph[p.second][i];
                        cost_t nextCost = min_cost[p.second] + e.cost + potential[p.second] - potential[e.to];
                        if (e.cap > 0 && min_cost[e.to] > nextCost) {
                            min_cost[e.to] = nextCost;
                            prevv[e.to] = p.second, preve[e.to] = i;
                            que.emplace(min_cost[e.to], e.to);
                        }
                    }
                }
                if (min_cost[t] == INF) return -1;
                for (int v = 0; v < V; v++) potential[v] += min_cost[v];
                flow_t addflow = f;
                for (int v = t; v != s; v = prevv[v]) {
                    addflow = std::min(addflow, graph[prevv[v]][preve[v]].cap);
                }
                f -= addflow;
                ret += addflow * potential[t];
                for (int v = t; v != s; v = prevv[v]) {
                    edge& e = graph[prevv[v]][preve[v]];
                    e.cap -= addflow;
                    graph[v][e.rev].cap += addflow;
                }
            }
            return ret;
        }

        void output() {
            for (int i = 0; i < graph.size(); i++) {
                for (auto& e : graph[i]) {
                    if (e.isrev) continue;
                    auto& rev_e = graph[e.to][e.rev];
                    cout << i << "->" << e.to << " (flow: " << rev_e.cap << "/" << rev_e.cap + e.cap << ")" << endl;
                }
            }
        }
    };

    struct Node {
        int id, i, j;
        Node(int id = -1, int i = -1, int j = -1) : id(id), i(i), j(j) {}
        std::string str() const {
            return format("Node [id=%d, p=(%d, %d)]", id, i, j);
        }
        friend std::ostream& operator<<(std::ostream& o, const Node& obj) {
            o << obj.str();
            return o;
        }
    };

    using Assign = std::pair<Node, Node>;

    struct Result {
        int total_cost;
        vector<vector<Assign>> color_to_assign;
    };

    vector<Assign> get_assign(
        const vector<Node>& S, const vector<Node>& T, const PrimalDual<int, int>& pd) {
        int ns = S.size();
        vector<Assign> assign;
        for (int u = 1; u <= (int)S.size(); u++) {
            for (const auto& e : pd.graph[u]) {
                if (e.isrev) continue;
                const auto& rev_e = pd.graph[e.to][e.rev];
                if (!rev_e.cap) continue;
                // i -> e.to
                int v = e.to;
                assign.emplace_back(S[u - 1], T[v - ns - 1]);
            }
        }
        return assign;
    }

    Result calc_assign(const Path& route, const vector<int>& perm) {
        Result result;
        result.color_to_assign.resize(C + 1);
        // create target board
        Board target(N + 2, vector<int>(N + 2, 0));
        vector<int> color_list;
        for (int c : perm) {
            for (int i = 0; i < g_count[c]; i++) {
                color_list.push_back(c);
            }
        }
        for (int n = 0; n < N * N; n++) {
            target[route[n].i][route[n].j] = color_list[n];
        }
        // create nodes
        vector<vector<Node>> S(C + 1), T(C + 1);
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                int sc = g_board[i][j];
                S[sc].emplace_back(S[sc].size() + 1, i, j);
            }
        }
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                int tc = target[i][j];
                T[tc].emplace_back(S[tc].size() + T[tc].size() + 1, i, j);
            }
        }
        // mincostflow
        int total_cost = 0;
        for (int c = 1; c <= C; c++) {
            int ns = S[c].size(), nt = T[c].size();
            int V = ns + nt + 2;
            PrimalDual<int, int> pd(V);
            // u=0 から v in 1..s に容量 1, コスト 0 の辺を張る
            for (const auto& v : S[c]) {
                pd.add_edge(0, v.id, 1, 0);
            }
            // u in 1..s から v in s+1...s+t に容量 inf, コスト dist(u, v) の辺を張る
            for (const auto& u : S[c]) {
                for (const auto& v : T[c]) {
                    int dist = abs(u.i - v.i) + abs(u.j - v.j);
                    pd.add_edge(u.id, v.id, pd.INF, dist);
                }
            }
            // u in s+1...s+t から v=s+t+1 に容量 1, コスト 0 の辺を張る
            for (const auto& v : T[c]) {
                pd.add_edge(v.id, V - 1, 1, 0);
            }
            //double elapsed = timer.elapsedMs();
            int cost = pd.min_cost_flow(0, V - 1, ns);
            //dump(c, cost, V, timer.elapsedMs() - elapsed);
            total_cost += cost;

            result.color_to_assign[c] = get_assign(S[c], T[c], pd);
        }

        result.total_cost = total_cost;

        return result;
    }

}

namespace NRoute2 {

    struct Node;
    using NodePtr = std::shared_ptr<Node>;
    struct Node {
        Point p; // 位置
        NodePtr other;
    };

    struct State {
        Board S; // 盤面
        NFlow::Result assign; // 割当て
        Path route; // 移動する順番
        vector<vector<bool>> fixed; // 移動終了したか？

        vector<Move> moves; // 移動履歴

        State(const NFlow::Result& assign, const Path& route)
            : S(g_board), assign(assign), route(route), fixed(N + 2, vector<bool>(N + 2, false)) {
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    fixed[i][j] = false;
                }
            }
        }

        void solve() {
            // route に沿って揃えていく
            vector<vector<NodePtr>> from_map(N + 2, vector<NodePtr>(N + 2, nullptr));
            vector<vector<NodePtr>> to_map(N + 2, vector<NodePtr>(N + 2, nullptr));
            for (int c = 1; c <= C; c++) {
                for (const auto& n2n : assign.color_to_assign[c]) {
                    const auto& from = n2n.first;
                    const auto& to = n2n.second;
                    NodePtr nfrom = std::make_shared<Node>();
                    NodePtr nto = std::make_shared<Node>();
                    nfrom->p = Point(from.i, from.j);
                    nfrom->other = nto;
                    nto->p = Point(to.i, to.j);
                    nto->other = nfrom;
                    from_map[nfrom->p.i][nfrom->p.j] = nfrom;
                    to_map[nto->p.i][nto->p.j] = nto;
                }
            }
            auto move_node = [&](NodePtr nto) {
                // ノード nfrom をノード nto に移動させる最短パス　移動禁止領域: fixed
                // 移動するたびにアップデートが必要なことに注意
                // TODO: 移動のバリエーションを考慮
                auto now_pos = nto->other->p;
                auto dst_pos = nto->p;
                int dist = now_pos.distance(dst_pos);
                while (now_pos != dst_pos) {
                    // なるべく他のセルを巻き込んで total_cost が 2 減るような方向を選びたい
                    // TODO: dist を減らさなくても total_cost が 2 減るような移動を優先する？
                    int min_cost_diff = 1000, min_cost_dir = -1;
                    for (int d = 0; d < 4; d++) {
                        auto next_pos = now_pos + dir[d];
                        int ndist = next_pos.distance(dst_pos);
                        if (!fixed[next_pos.i][next_pos.j] && ndist < dist) {
                            // 大前提として、着目しているマスの到達距離は減る
                            // next_pos -> now_pos の移動が到達距離を縮めるか？
                            NodePtr from = from_map[next_pos.i][next_pos.j];
                            NodePtr to = from->other;
                            int dist2 = from->p.distance(to->p);
                            int ndist2 = now_pos.distance(to->p);
                            int diff = ndist2 - dist2 + ndist - dist;
                            if (diff < min_cost_diff) {
                                min_cost_diff = diff;
                                min_cost_dir = d;
                            }
                        }
                    }
                    int d = min_cost_dir;
                    auto next_pos = now_pos + dir[d];
                    int ndist = next_pos.distance(dst_pos);
                    if (!fixed[next_pos.i][next_pos.j] && ndist < dist) {
                        // 採用: now_pos, next_pos を swap する
                        // 点の swap
                        std::swap(from_map[now_pos.i][now_pos.j]->p, from_map[next_pos.i][next_pos.j]->p);
                        // pointer の swap
                        std::swap(from_map[now_pos.i][now_pos.j], from_map[next_pos.i][next_pos.j]);
                        // 盤面の swap
                        std::swap(S[now_pos.i][now_pos.j], S[next_pos.i][next_pos.j]);
                        // !!!異なる色ならば!!!、moves に反映
                        if (S[now_pos.i][now_pos.j] != S[next_pos.i][next_pos.j]) {
                            moves.emplace_back(now_pos.i, now_pos.j, next_pos.i, next_pos.j);
                        }
                        // assign
                        now_pos = next_pos; dist = now_pos.distance(dst_pos);
                    }
                }
                // 到着したので fix
                fixed[dst_pos.i][dst_pos.j] = true;
            };
            for (int idx = 0; idx < N * N; idx++) {
                NodePtr nto = to_map[route[idx].i][route[idx].j];
                move_node(nto);
            }
        }

        std::string str() const {
            std::ostringstream oss;
            oss << "--- State ---\n"
                << "N = " << N << ", C = " << C << '\n';
            for (int i = 0; i < N + 2; i++) {
                for (int j = 0; j < N + 2; j++) {
                    oss << S[i][j] << ' ';
                }
                oss << '\n';
            }
            oss << "-------------\n";
            return oss.str();
        }

        friend std::ostream& operator<<(std::ostream& o, const State& obj) {
            o << obj.str();
            return o;
        }

        void output(std::ostream& o) const {
            o << moves.size() << '\n';
            for (const auto& t : moves) {
                int i1, j1, i2, j2;
                std::tie(i1, j1, i2, j2) = t;
                o << i1 - 1 << ' ' << j1 - 1 << ' ' << i2 - 1 << ' ' << j2 - 1 << '\n';
            }
        }
    };

}

//#define LOCAL_MODE

int main() {
    std::ios::sync_with_stdio(false);
    cin.tie(0);

#ifdef LOCAL_MODE
    std::ifstream ifs("C:\\dev\\TCMM\\problems\\MM128\\in\\5.in");
    std::istream& in = ifs;
    std::ofstream ofs("C:\\dev\\TCMM\\problems\\MM128\\out\\5.out");
    std::ostream& out = ofs;
#else
    std::istream& in = cin;
    std::ostream& out = cout;
#endif

    init(in);

    auto spiral = generate_spiral(N); // 移動は常に spiral
    vector<int> perm;
    for (int c = 1; c <= C; c++) perm.push_back(c);
    // 螺旋状に 11..22..33..CC を配置したときのコストを求めたい
    double elapsed = timer.elapsedMs();
    auto assign = NFlow::calc_assign(spiral, perm);
    dump(assign.total_cost, timer.elapsedMs() - elapsed);

    // 順列変更山登り
    auto best_assign(assign);
    vector<int> best_perm = perm;
    while (timer.elapsedMs() < 5000) {
        int i = rnd.next_int(C), j;
        do {
            j = rnd.next_int(C);
        } while (i == j);
        std::swap(perm[i], perm[j]);
        assign = NFlow::calc_assign(spiral, perm);
        if (assign.total_cost < best_assign.total_cost) {
            best_assign = assign;
            best_perm = perm;
            dump(best_assign.total_cost, timer.elapsedMs());
        }
    }

    NRoute2::State state(best_assign, spiral);
    state.solve();

    dump(state.moves.size());

    state.output(out);

    return 0;
}