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
    unsigned int frame_mask = (1ULL << (N + 2)) - 1;
    unsigned int inner_mask = (1ULL << (N + 1)) | 1;
    g_board_mask[0] = g_board_mask[N + 1] = frame_mask;
    for (int i = 1; i <= N; i++) g_board_mask[i] = inner_mask;
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

namespace NFlow {

    struct PrimalDual {
        const int INF;

        struct edge {
            int to;
            int cap;
            int cost;
            int rev;
            bool isrev;
            edge(int to = -1, int cap = -1, int cost = -1, int rev = -1, bool isrev = false)
                : to(to), cap(cap), cost(cost), rev(rev), isrev(isrev) {}
        };

        vector<vector<edge>> graph;
        vector<int> potential, min_cost;
        vector<int> prevv, preve;

        PrimalDual(int V) : INF(std::numeric_limits<int>::max()), graph(V) {}

        void add_edge(int from, int to, int cap, int cost) {
            graph[from].emplace_back(to, cap, cost, (int)graph[to].size(), false);
            graph[to].emplace_back(from, 0, -cost, (int)graph[from].size() - 1, true);
        }

        int min_cost_flow(int s, int t, int f) {
            int V = (int)graph.size();
            int ret = 0;
            using Pi = ll;
            std::priority_queue<Pi, vector<Pi>, std::greater<Pi>> que;
            potential.assign(V, 0);
            preve.assign(V, -1);
            prevv.assign(V, -1);

            while (f > 0) {
                min_cost.assign(V, INF);
                que.emplace(s);
                min_cost[s] = 0;
                while (!que.empty()) {
                    Pi p = que.top(); que.pop();
                    int pf = p >> 32, ps = p & 0xFFFFFFFFLL;
                    if (min_cost[ps] < pf) continue;
                    for (int i = 0; i < (int)graph[ps].size(); i++) {
                        edge& e = graph[ps][i];
                        int nextCost = min_cost[ps] + e.cost + potential[ps] - potential[e.to];
                        if (e.cap > 0 && min_cost[e.to] > nextCost) {
                            min_cost[e.to] = nextCost;
                            prevv[e.to] = ps, preve[e.to] = i;
                            que.emplace(((ll)min_cost[e.to] << 32) | e.to);
                        }
                    }
                }
                if (min_cost[t] == INF) return -1;
                for (int v = 0; v < V; v++) potential[v] += min_cost[v];
                int addflow = f;
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
        const vector<Node>& S, const vector<Node>& T, const PrimalDual& pd) {
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
            PrimalDual pd(V);
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

namespace NSolver {

    constexpr int dy[] = { 0, -1, 0, 1 };
    constexpr int dx[] = { 1, 0, -1, 0 };
    constexpr int dy8[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    constexpr int dx8[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    // 連結成分数差分計算高速化のためのルックアップテーブル
    constexpr int lut[] = {
      -1, 0,-1, 0, 0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
      -1, 0,-1, 0, 0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
       0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
      -1, 0,-1, 0, 0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
      -1, 0,-1, 0, 0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
       0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
       0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    }; // 0: 連結成分変化なし、-1: 連結成分 1 減少、1: 連結成分要調査

    // 連結性判定用
    // TODO: UnionFind 速度検証・打ち切り付き差分計算BFS・輪郭追跡etc...
    using cc_type = int;
    struct CC {

        vector<vector<cc_type>> T;

        inline int get_lut_flag(int i, int j, cc_type c) const {
            // 8 近傍を調べる
            int mask = 0;
            for (int d = 0; d < 8; d++) {
                mask |= (int(c == T[i + dy8[d]][j + dx8[d]]) << d);
            }
            return lut[mask];
        }

        bool can_swap(int i1, int j1, int i2, int j2) {
            if (T[i1][j1] == T[i2][j2]) return true;
            int a = T[i1][j1], b = T[i2][j2];
            bool a_from = false, a_to = false, b_from = false, b_to = false;

            // a_to: T[i2][j2] の 4 近傍のどれか一つは a
            if (g_count[a] == 1) {
                // 1 要素のみならどこに移動しても構わない
                a_from = a_to = true;
            }
            else {
                for (int d = 0; d < 4; d++) {
                    if (T[i2 + di[d]][j2 + dj[d]] == a) {
                        a_to = true; break;
                    }
                }
            }
            if (!a_to) return false;

            // b_to: T[i1][j1] の 4 近傍のどれか一つは b
            if (g_count[b] == 1) {
                b_from = b_to = true;
            }
            else {
                for (int d = 0; d < 4; d++) {
                    if (T[i1 + di[d]][j1 + dj[d]] == b) {
                        b_to = true; break;
                    }
                }
            }
            if (!b_to) return false;

            // do swap
            std::swap(T[i1][j1], T[i2][j2]);
            
            // a_from: T[i1][j1] の 8 近傍に含まれる a は、一つの連結成分に含まれる
            if (!a_from) {
                if (get_lut_flag(i1, j1, a) && !is_valid_cc(i1, j1, a)) {
                    std::swap(T[i1][j1], T[i2][j2]);
                    return false;
                }
            }

            // b_from
            // T[i2][j2] の 8 近傍に含まれる b は、一つの連結成分に含まれる
            if (!b_from) {
                if (get_lut_flag(i2, j2, b) && !is_valid_cc(i2, j2, b)) {
                    std::swap(T[i1][j1], T[i2][j2]);
                    return false;
                }
            }

            std::swap(T[i1][j1], T[i2][j2]);
            return true;
        }

        bool is_valid_cc(int i, int j, int c) const {
            // (i, j) の周り 4 近傍の色 c のセルが全て同一連結成分上にあるか判定する
            static unsigned int bmask[32];
            memcpy(bmask, g_board_mask, sizeof(unsigned int) * 32);
            int si, sj, cnt = 0;
            {
                int sd;
                for (sd = 0; sd < 4; sd++) if (T[i + di[sd]][j + dj[sd]] == c) break;
                si = i + di[sd]; sj = j + dj[sd];
            }
            fqu.reset();
            fqu.push((si << 5) | sj);
            bmask[si] |= (1U << sj);
            cnt++;
            while (!fqu.empty()) {
                int cij = fqu.pop(), ci = (cij >> 5), cj = (cij & 0b11111);
                for (int d = 0; d < 4; d++) {
                    int ni = ci + di[d], nj = cj + dj[d];
                    if ((bmask[ni] & (1U << nj)) || T[ni][nj] != c) continue;
                    fqu.push((ni << 5) | nj);
                    bmask[ni] |= (1U << nj);
                    cnt++;
                }
            }
            return cnt == g_count[c];
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
                            if ((bmask[ni] & (1U << nj)) || T[ni][nj] != c) continue;
                            //if (((bmask[ni] >> nj) & 1) || T[ni][nj] != c) continue;
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
    };

    struct Node;
    using NodePtr = std::shared_ptr<Node>;
    struct Node {
        Point p; // 位置
        int c;   // 色
        NodePtr other;

        std::string str() const {
            return format("Node [this=%p, p=%s, c=%d, other=%p, other->p=%s, other->c=%d]",
                this, p.str().c_str(), c, other.get(), other->p.str().c_str(), other->c);
        }
        friend std::ostream& operator<<(std::ostream& o, const Node& obj) {
            o << obj.str();
            return o;
        }
        friend std::ostream& operator<<(std::ostream& o, const NodePtr& obj) {
            o << obj->str();
            return o;
        }
    };

    struct Trans {
        enum struct Type { BOARD, TARGET };
        Type type;
        int i1, j1, i2, j2;
        int diff;
        bool is_valid;

        std::string str() const {
            return format("Trans [type=%d, i1=%d, j1=%d, i2=%d, j2=%d, diff=%d, is_valid=%d]", type, i1, j1, i2, j2, diff, is_valid);
        }
        friend std::ostream& operator<<(std::ostream& o, const Trans& obj) {
            o << obj.str();
            return o;
        }
    };

    struct State {
        NFlow::Result assign; // 割当て
        Path route; // 移動する順番
        vector<vector<bool>> fixed; // 移動終了したか？

        vector<vector<NodePtr>> board; // 目的地情報付き現在地
        vector<vector<NodePtr>> target;   // 現在地情報付き目的地

        CC conn;

        vector<Move> moves; // 移動履歴
        int total_distance;

        // 2 つの遷移を考える

        // 1. 4-adjacent な 2 点の現在地を入れ替える
        // -> move に対応　移動コストが 1 増える

        // 2. 任意の 2 点の目的地を入れ替える
        // -> 完成盤面をいじる操作
        // -> 移動コストが生じない（!）
        // -> 連結成分条件を保つ必要がある

        State(const NFlow::Result& assign, const Path& route) :
            assign(assign), route(route),
            fixed(N + 2, vector<bool>(N + 2, true)),
            board(N + 2, vector<NodePtr>(N + 2, nullptr)),
            target(N + 2, vector<NodePtr>(N + 2, nullptr)),
            total_distance(assign.total_cost) {
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    fixed[i][j] = false;
                }
            }

            //memset(conn.T, 0, sizeof(cc_type) * (N + 2) * (N + 2));
            conn.T.resize(N + 2, vector<cc_type>(N + 2, 0));
            for (int c = 1; c <= C; c++) {
                for (const auto& n2n : assign.color_to_assign[c]) {
                    const auto& from = n2n.first;
                    const auto& to = n2n.second;
                    NodePtr nfrom = std::make_shared<Node>();
                    NodePtr nto = std::make_shared<Node>();
                    int c = g_board[from.i][from.j];
                    nfrom->p = Point(from.i, from.j);
                    nfrom->c = c;
                    nfrom->other = nto;
                    nto->p = Point(to.i, to.j);
                    nto->c = c;
                    nto->other = nfrom;
                    board[nfrom->p.i][nfrom->p.j] = nfrom;
                    target[nto->p.i][nto->p.j] = nto;
                    conn.T[nto->p.i][nto->p.j] = nto->c;
                }
            }
        }

        Trans can_board_swap(int i1, int j1, int i2, int j2) const {
            Trans t;
            t.type = Trans::Type::BOARD;
            if (abs(i1 - i2) + abs(j1 - j2) != 1) {
                t.is_valid = false;
                return t;
            }
            t.i1 = i1; t.j1 = j1; t.i2 = i2; t.j2 = j2;
            t.diff
                = board[i1][j1]->p.distance(board[i2][j2]->other->p)
                + board[i2][j2]->p.distance(board[i1][j1]->other->p)
                - board[i1][j1]->p.distance(board[i1][j1]->other->p)
                - board[i2][j2]->p.distance(board[i2][j2]->other->p);
            t.is_valid = true;
            return t;
        }

        Trans can_board_swap(const Point& p1, const Point& p2) const {
            return can_board_swap(p1.i, p1.j, p2.i, p2.j);
        }

        void board_swap(int i1, int j1, int i2, int j2) {
            int dist =
                board[i1][j1]->p.distance(board[i1][j1]->other->p)
                + board[i2][j2]->p.distance(board[i2][j2]->other->p);
            // pointer の swap
            std::swap(board[i1][j1], board[i2][j2]);
            // 点の swap
            std::swap(board[i1][j1]->p, board[i2][j2]->p);
            // !!!異なる色ならば!!!、moves に反映
            if (board[i1][j1]->c != board[i2][j2]->c) {
                moves.emplace_back(i1, j1, i2, j2);
            }
            int ndist =
                board[i1][j1]->p.distance(board[i1][j1]->other->p)
                + board[i2][j2]->p.distance(board[i2][j2]->other->p);
            total_distance += ndist - dist;
        }

        void board_swap(const Point& p1, const Point& p2) {
            return board_swap(p1.i, p1.j, p2.i, p2.j);
        }

        void board_swap(const Trans& t) {
            std::swap(board[t.i1][t.j1], board[t.i2][t.j2]);
            std::swap(board[t.i1][t.j1]->p, board[t.i2][t.j2]->p);
            if (board[t.i1][t.j1]->c != board[t.i2][t.j2]->c) {
                moves.emplace_back(t.i1, t.j1, t.i2, t.j2);
            }
            total_distance += t.diff;
        }

        Trans can_target_swap(int i1, int j1, int i2, int j2) {
            Trans t;
            t.type = Trans::Type::TARGET;
            if (conn.can_swap(i1, j1, i2, j2)) {
                t.i1 = i1; t.j1 = j1; t.i2 = i2; t.j2 = j2;
                t.diff
                    = target[i1][j1]->p.distance(target[i2][j2]->other->p)
                    + target[i2][j2]->p.distance(target[i1][j1]->other->p)
                    - target[i1][j1]->p.distance(target[i1][j1]->other->p)
                    - target[i2][j2]->p.distance(target[i2][j2]->other->p);
                t.is_valid = true;
                return t;
            }
            else {
                t.is_valid = false;
                return t;
            }
        }

        Trans can_target_swap(const Point& p1, const Point& p2) {
            return can_target_swap(p1.i, p1.j, p2.i, p2.j);
        }

        void target_swap(int i1, int j1, int i2, int j2) {
            int dist =
                target[i1][j1]->p.distance(target[i1][j1]->other->p)
                + target[i2][j2]->p.distance(target[i2][j2]->other->p);
            // pointer の swap
            std::swap(target[i1][j1], target[i2][j2]);
            // 点の swap
            std::swap(target[i1][j1]->p, target[i2][j2]->p);
            int ndist =
                target[i1][j1]->p.distance(target[i1][j1]->other->p)
                + target[i2][j2]->p.distance(target[i2][j2]->other->p);
            total_distance += ndist - dist;
        }

        void target_swap(const Point& p1, const Point& p2) {
            return target_swap(p1.i, p1.j, p2.i, p2.j);
        }

        void target_swap(const Trans& t) {
            std::swap(target[t.i1][t.j1], target[t.i2][t.j2]);
            std::swap(target[t.i1][t.j1]->p, target[t.i2][t.j2]->p);
            std::swap(conn.T[t.i1][t.j1], conn.T[t.i2][t.j2]);
            total_distance += t.diff;
        }

        void move_node(NodePtr nto) {
            // ノード nfrom をノード nto に最短パスで移動させる　移動禁止領域: fixed
            // 必ずしもこの時点での nfrom を nto に移動させる必要はない？
            // pointer をすげ替えてコストが減るなら変更すべき？
            // TODO: 移動のバリエーションを考慮　極小 dfs / beamsearch とか？
            auto now_pos = nto->other->p;
            auto dst_pos = nto->p;
            int dist = now_pos.distance(dst_pos);
            // TODO: 無駄な動きがいくつか見られる: seed 17 とか？
            while (now_pos != dst_pos) {
                // nto の色が既に揃っている場合
                if (board[dst_pos.i][dst_pos.j]->c == nto->c) {
                    // nto の役割と from_map[dst_pos.i][dst_pos.j]->other の役割をチェンジ
                    target_swap(nto->p, board[dst_pos.i][dst_pos.j]->other->p);
                    fixed[dst_pos.i][dst_pos.j] = true;
                    return;
                }
                // なるべく他のセルを巻き込んで total_cost が 2 減るような方向を選びたい
                Trans trans; trans.diff = 1000;
                int min_cost_dir = -1;
                for (int d = 0; d < 4; d++) {
                    auto next_pos = now_pos + dir[d];
                    int ndist = next_pos.distance(dst_pos);
                    if (!fixed[next_pos.i][next_pos.j] && ndist < dist) {
                        // 大前提として、着目しているセルの到達距離は減る
                        // 交換対象となるセルの到達距離も縮むならなお良い
                        Trans t = can_board_swap(now_pos, next_pos);
                        if (t.is_valid && t.diff < trans.diff) {
                            trans = t;
                            min_cost_dir = d;
                        }
                    }
                }
                assert(min_cost_dir != -1);
                board_swap(trans);
                //cerr << total_distance << ' ' << moves.size() << '\n';
                // assign
                now_pos += dir[min_cost_dir]; dist = now_pos.distance(dst_pos);
                //vis(1);
            }
            // 到着したので fix
            fixed[dst_pos.i][dst_pos.j] = true;
        }

        void solve() {

            auto get_temp = [](double start_temp, double end_temp, double t, double T) {
                return end_temp + (start_temp - end_temp) * (T - t) / T;
            };

            // TODO: ノーコストの board swap も入れる -> あんまよくないかも
            // TODO: 遷移率の向上
            int loop = 0, valid = 0, accepted = 0;
            double start_time = timer.elapsedMs(), now_time, end_time = 7500;
            while ((now_time = timer.elapsedMs()) < end_time) {
                loop++;
                if (!(loop & 0xFFFFF)) {
                    dump(loop, valid, accepted, total_distance);
                    //vis(1);
                }

                int i1 = rnd.next_int(N) + 1, j1 = rnd.next_int(N) + 1, i2, j2;
                do {
                    i2 = rnd.next_int(N) + 1;
                    j2 = rnd.next_int(N) + 1;
                } while (i1 == i2 && j1 == j2);

                Trans t = can_target_swap(i1, j1, i2, j2);

                if (!t.is_valid) continue;

                valid++;

                double temp = get_temp(2.0, 0.0, now_time - start_time, end_time - start_time);
                double prob = exp(-t.diff / temp);

                if (rnd.next_double() < prob) {
                    accepted++;
                    target_swap(t);
                }

            }
            dump("annealing", loop, valid, accepted, total_distance, timer.elapsedMs());

            // route に沿って揃えていく
            // TODO: route のアレンジ
            for (int idx = 0; idx < N * N; idx++) {
                NodePtr nto = target[route[idx].i][route[idx].j];
                move_node(nto);
                if (false) {
                    // board swap
                    for (int i = 1; i <= N; i++) {
                        for (int j = 1; j < N; j++) {
                            if (board[i][j]->c == board[i][j + 1]->c) {
                                Trans t = can_board_swap(i, j, i, j + 1);
                                assert(t.is_valid);
                                if (t.diff < 0) {
                                    board_swap(t);
                                }
                            }
                        }
                    }
                    for (int i = 1; i < N; i++) {
                        for (int j = 1; j <= N; j++) {
                            if (board[i][j]->c == board[i + 1][j]->c) {
                                Trans t = can_board_swap(i, j, i + 1, j);
                                assert(t.is_valid);
                                if (t.diff < 0) {
                                    board_swap(t);
                                }
                            }
                        }
                    }
                }
                if (true) {
                    // target swap でコストが減らせるなら愚直に減らす
                    vector<vector<Point>> pts(C + 1);
                    for (int i = 1; i <= N; i++) {
                        for (int j = 1; j <= N; j++) {
                            if (fixed[i][j]) continue;
                            pts[target[i][j]->c].emplace_back(i, j);
                        }
                    }
                    for (int c = 1; c <= C; c++) {
                        int np = pts[c].size();
                        for (int i = 0; i < np - 1; i++) {
                            for (int j = i + 1; j < np; j++) {
                                Trans t = can_target_swap(pts[c][i], pts[c][j]);
                                if (t.diff < 0) {
                                    target_swap(t);
                                    // cerr << t << endl;
                                    // vis(1);
                                }
                            }
                        }
                    }
                }
            }
            dump("routing", timer.elapsedMs());
        }

        std::string str() const {
            std::ostringstream oss;
            oss << "--- State ---\n"
                << "N = " << N << ", C = " << C << '\n';
            for (int i = 0; i < N + 2; i++) {
                for (int j = 0; j < N + 2; j++) {
                    oss << board[i][j]->c << ' ';
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

#ifdef _MSC_VER
        void vis(int delay = 0) {
            int grid_size = 960 / (N + 2);
            int height = grid_size * (N + 2), width = grid_size * (N + 2);
            cv::Mat_<cv::Vec3b> img(height, width, cv::Vec3b(255, 255, 255));
            // to_map
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    cv::Rect roi(grid_size * j, grid_size * i, grid_size, grid_size);
                    cv::rectangle(img, roi, g_color[target[i][j]->c], cv::FILLED);
                }
            }
            // grid line
            for (int i = 1; i < N + 2; i++) {
                cv::line(img, cv::Point(0, i * grid_size), cv::Point((N + 2) * grid_size, i * grid_size), cv::Scalar(0, 0, 0), 1);
                cv::line(img, cv::Point(i * grid_size, 0), cv::Point(i * grid_size, (N + 2) * grid_size), cv::Scalar(0, 0, 0), 1);
            }
            // from_map
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    cv::Point p(grid_size * (j + 0.5), grid_size * (i + 0.5));
                    cv::circle(img, p, grid_size / 3, g_color[board[i][j]->c], cv::FILLED);
                    auto color = (board[i][j]->p == board[i][j]->other->p) ? cv::Scalar(0, 255, 255) : cv::Scalar(50, 50, 50);
                    cv::circle(img, p, grid_size / 3, color, 1);
                }
            }
            // prev_move
            if (!moves.empty()) {
                int i1, j1, i2, j2; std::tie(i1, j1, i2, j2) = moves.back();
                cv::Rect roi1(grid_size * j1, grid_size * i1, grid_size, grid_size);
                cv::Rect roi2(grid_size * j2, grid_size * i2, grid_size, grid_size);
                cv::rectangle(img, roi1, cv::Scalar(0, 255, 255), 3);
                cv::rectangle(img, roi2, cv::Scalar(0, 255, 255), 3);
            }
            // arrow
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    auto pf = board[i][j]->p;
                    cv::Point cvpf(grid_size * (pf.j + 0.5), grid_size * (pf.i + 0.5));
                    auto pt = board[i][j]->other->p;
                    cv::Point cvpt(grid_size * (pt.j + 0.5), grid_size * (pt.i + 0.5));
                    cv::arrowedLine(img, cvpf, cvpt, cv::Scalar(0, 255, 255), 1, 8, 0, 0.1);
                }
            }
            cv::imshow("img", img);
            cv::waitKey(delay);
        }
#endif

    };

}

//#define LOCAL_MODE

int main() {
    std::ios::sync_with_stdio(false);
    cin.tie(0);

#ifdef LOCAL_MODE
    std::ifstream ifs("C:\\dev\\TCMM\\problems\\MM128\\in\\1.in");
    std::istream& in = ifs;
    std::ofstream ofs("C:\\dev\\TCMM\\problems\\MM128\\out\1.out");
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

    dump("first assign", assign.total_cost, timer.elapsedMs() - elapsed);

    // 順列変更山登り
    auto best_assign(assign);
    vector<int> best_perm = perm;
    int loop = 0;
    while (timer.elapsedMs() < 3000) {
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
        loop++;
    }
    dump("climbing assign", best_assign.total_cost, loop, timer.elapsedMs());

    NSolver::State state(best_assign, spiral);

    //state.vis();

    state.solve();

    state.output(out);

    dump("answer", state.moves.size(), timer.elapsedMs());

    return 0;
}