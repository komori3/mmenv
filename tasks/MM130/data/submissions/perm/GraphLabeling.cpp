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
template<typename A, size_t N, typename T> void Fill(A(&array)[N], const T& val) { std::fill((T*)array, (T*)(array + N), val); }
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
    int n = (int)v.size();
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
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }



using namespace std;

int num_nodes;
int num_edges;
vector<pii> edges;
vector<vector<bool>> adjmat;
vector<vector<int>> adjlist;

void init(istream& in) {
    in >> num_nodes;
    in >> num_edges;
    edges.resize(num_edges);
    in >> edges;
    adjmat.resize(num_nodes, vector<bool>(num_nodes, false));
    adjlist.resize(num_nodes);
    for (const auto& edge : edges) {
        int u, v; std::tie(u, v) = edge;
        adjmat[u][v] = adjmat[v][u] = true;
        adjlist[u].push_back(v);
        adjlist[v].push_back(u);
    }
}

struct State {
    static constexpr int MAX = 3000000;

    int score;
    vector<int> node_vals;

    bool* invalid;
    bool* used_node_val;
    bool* used_edge_val;

    State() { 
        invalid = new bool[MAX];
        used_node_val = new bool[MAX];
        used_edge_val = new bool[MAX];
        reset();
    }

    void reset() {
        score = 0;
        node_vals = vector<int>(num_nodes, -1);
        memset(invalid, 0, sizeof(bool) * MAX);
        memset(used_node_val, 0, sizeof(bool) * MAX);
        memset(used_edge_val, 0, sizeof(bool) * MAX);
    }

    void solve(int start_node) {
        queue<int> qu;
        node_vals[start_node] = 0;
        used_node_val[0] = true;
        qu.push(start_node);
        while (!qu.empty()) {
            int u = qu.front(); qu.pop();
            for (int v : adjlist[u]) {
                if (node_vals[v] != -1) continue; // visited
                // v に隣接するノードで、used のものの値を取ってくる
                vector<int> xs;
                for (int w : adjlist[v]) if (node_vals[w] != -1) xs.push_back(node_vals[w]);
                sort(xs.rbegin(), xs.rend());

                memset(invalid, 0, sizeof(bool) * MAX);
                for (int i = 0; i < (int)xs.size(); i++) {
                    for (int j = i; j < (int)xs.size(); j++) {
                        if ((xs[i] + xs[j]) % 2 == 0) {
                            invalid[(xs[i] + xs[j]) / 2] = true;
                        }
                    }
                }

                //int nx = xs.front() + 1;
                int nx = 0;
                for (;; nx++) {
                    if (used_node_val[nx]) continue;
                    bool ok = true;
                    for (int x : xs) {
                        if (used_edge_val[abs(nx - x)] | invalid[nx]) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) break;
                }
                for (int x : xs) used_edge_val[abs(nx - x)] = true;
                used_node_val[nx] = true;
                node_vals[v] = nx;
                qu.push(v);
            }
        }
        score = *max_element(node_vals.begin(), node_vals.end());
    }

    void solve2(const vector<int>& perm) {
        // bfs である必要はない
        for (int u : perm) {
            // u に隣接するノードで、used のものの値を取ってくる
            vector<int> adj_node_vals;
            for (int v : adjlist[u]) if (node_vals[v] != -1) adj_node_vals.push_back(node_vals[v]);
            sort(adj_node_vals.begin(), adj_node_vals.end());
            memset(invalid, 0, sizeof(bool) * MAX);
            for (int i = 0; i < (int)adj_node_vals.size(); i++) {
                for (int j = i; j < (int)adj_node_vals.size(); j++) {
                    if ((adj_node_vals[i] + adj_node_vals[j]) % 2 == 0) {
                        invalid[(adj_node_vals[i] + adj_node_vals[j]) / 2] = true;
                    }
                }
            }
            int node_val = 0;
            for (;; node_val++) {
                if (used_node_val[node_val]) continue;
                bool ok = true;
                for (int adj_node_val : adj_node_vals) {
                    if (used_edge_val[abs(node_val - adj_node_val)] | invalid[node_val]) {
                        ok = false;
                        break;
                    }
                }
                if (ok) break;
            }
            for (int adj_node_val : adj_node_vals) {
                used_edge_val[abs(node_val - adj_node_val)] = true;
            }
            used_node_val[node_val] = true;
            node_vals[u] = node_val;
        }
        score = *max_element(node_vals.begin(), node_vals.end());
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    //ifstream ifs("in/69.in");
    //istream& in = ifs;
    istream& in = cin;

    init(in);

    dump(num_nodes, num_edges);

    State state;
    int best_score = INT_MAX;
    int start_node = 0;
    int loop = 0;
    vector<int> ans;
    vector<int> perm(num_nodes);
    for (int i = 0; i < num_nodes; i++) perm[i] = i;

    while (true) {
        double rap_start = timer.elapsed_ms();

        state.solve2(perm);

        if (chmin(best_score, state.score)) {
            ans = state.node_vals;
            dump(best_score);
        }

        loop++;

        double rap_end = timer.elapsed_ms();
        double rap_time = rap_end - rap_start;
        if (timer.elapsed_ms() + rap_time > 9000) break;

        shuffle_vector(perm, rnd);
        state.reset();
    }

    dump(loop, best_score);

    ostringstream out;
    for (int x : ans) {
        out << x << " ";
    }

    cout << out.str() << endl;
    cout.flush();

    return 0;
}