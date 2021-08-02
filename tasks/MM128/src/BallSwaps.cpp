//#define NDEBUG
#include "bits/stdc++.h"
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#ifdef _MSC_VER
#include <ppl.h>
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

using Path = std::vector<Point>;
using Board = std::vector<std::vector<int>>;
using Move = std::tuple<int, int, int, int>;

namespace NSpiral {
    struct State {

        int N, C;
        vector<int> cnt;
        Board board;
        vector<vector<bool>> fixed;

        vector<Move> moves;

        State(std::istream& in) {
            in >> N >> C;
            board.resize(N, vector<int>(N));
            in >> board;
            fixed.resize(N, vector<bool>(N, false));
            cnt.resize(C, 0);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    cnt[board[i][j]]++;
                }
            }
        }

        inline bool is_inside(const Point& p) const {
            return 0 <= p.i && p.i < N && 0 <= p.j && p.j < N;
        }

        Path get_shortest_path(int x, const Point& dst) const {
            //// dst に最も近い数字 x の場所を求めて、経路復元
            int mindist = 1e9;
            Point src;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    if (fixed[i][j] || board[i][j] != x) continue;
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
                    if (is_inside(p) && !fixed[p.i][p.j] && dist < mindist) {
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
                std::swap(board[p1.i][p1.j], board[p2.i][p2.j]);
                moves.emplace_back(p1.i, p1.j, p2.i, p2.j);
            }
        }

        void solve(const vector<int>& perm) {
            // 螺旋状のターゲットを作成
            vector<int> target;
            Path spiral;
            {
                vector<vector<bool>> visited(N, vector<bool>(N, false));
                Point p(0, 0);
                int d = 0;
                spiral.push_back(p);
                visited[p.i][p.j] = true;
                while (true) {
                    if (is_inside(p + dir[d]) && !visited[p.i + dir[d].i][p.j + dir[d].j]) {
                        p += dir[d];
                        spiral.push_back(p);
                        visited[p.i][p.j] = true;
                    }
                    else if (is_inside(p + dir[(d + 1) & 3]) && !visited[p.i + dir[(d + 1) & 3].i][p.j + dir[(d + 1) & 3].j]) {
                        d = (d + 1) & 3;
                        p += dir[d];
                        spiral.push_back(p);
                        visited[p.i][p.j] = true;
                    }
                    else {
                        break;
                    }
                }
            }
            for (int i = 0; i < N; i++) {
                if (i % 2 == 0) {
                    for (int j = 0; j < N; j++) {
                        spiral.emplace_back(i, j);
                    }
                }
                else {
                    for (int j = N - 1; j >= 0; j--) {
                        spiral.emplace_back(i, j);
                    }
                }
            }

            for (int c : perm) {
                for (int i = 0; i < cnt[c]; i++) {
                    target.push_back(c);
                }
            }

            // 蛇腹状のパスに沿って揃えていく
            // elems[idx] != target[idx] となるような idx に対して
            // 既に揃えられたマス以外を通って数字 target[idx] を zigzag[idx] に移動させるような最短パスを求める
            // 関数: ある数字 x をセル c に移動させる最短パス　移動禁止領域: fixed
            for (int idx = 0; idx < N * N; idx++) {
                int i = spiral[idx].i, j = spiral[idx].j;
                if (board[i][j] == target[idx]) {
                    fixed[i][j] = true;
                    continue;
                }
                auto path = get_shortest_path(target[idx], spiral[idx]);
                do_moves(path);
                fixed[i][j] = true;
            }

        }

        std::string str() const {
            std::ostringstream oss;
            oss << "--- State ---\n"
                << "N = " << N << ", C = " << C << '\n';
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    oss << board[i][j] << ' ';
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
                o << i1 << ' ' << j1 << ' ' << i2 << ' ' << j2 << '\n';
            }
        }
    };
}

namespace NVariance {

    struct State {
        int N, C;
        vector<int> cnt;
        Board board;
        vector<Move> moves;

        State(std::istream& in) {
            in >> N >> C;
            board.resize(N, vector<int>(N));
            in >> board;
            cnt.resize(C, 0);
            vector<double> imean(C, 0.0);
            vector<double> jmean(C, 0.0);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    cnt[board[i][j]]++;
                    imean[board[i][j]] += i;
                    jmean[board[i][j]] += j;
                }
            }

            for (int c = 0; c < C; c++) {
                imean[c] /= cnt[c];
                jmean[c] /= cnt[c];
            }

            dump(C);
            dump(imean);
            dump(jmean);
        }
    };

}

//#define LOCAL_MODE

int main() {
    std::ios::sync_with_stdio(false);
    cin.tie(0);

#ifdef LOCAL_MODE
    std::ifstream ifs("C:\\dev\\TCMM\\problems\\MM128\\in\\2.in");
    std::istream& in = ifs;
    std::ofstream ofs("C:\\dev\\TCMM\\problems\\MM128\\out\\2.out");
    std::ostream& out = ofs;
#else
    std::istream& in = cin;
    std::ostream& out = cout;
#endif

    using namespace NSpiral;

    State init_state(in);

    int best_score = INT_MAX;
    State best_state(init_state);

    vector<int> perm(init_state.C);
    for (int i = 0; i < init_state.C; i++) perm[i] = i;

    int loop = 0;
    while (true) {
        State state(init_state);
        state.solve(perm);
        if (state.moves.size() < best_score) {
            dump(state.moves.size());
            best_score = state.moves.size();
            best_state = state;
        }
        shuffle_vector(perm, rnd);
        loop++;
        break; // only one iter
    }
    
    best_state.output(out);
    out.flush();

    dump(loop, timer.elapsedMs());

    return 0;
}