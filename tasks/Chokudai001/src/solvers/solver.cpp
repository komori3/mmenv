//#define NDEBUG
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include "bits/stdc++.h"
#include <unordered_map>
#include <unordered_set>
#include <random>
// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
/* const */
constexpr double PI = 3.141592653589793238462643;
/* io */
namespace aux {
    template<typename T, unsigned N, unsigned L> struct tp { static void print(std::ostream& os, const T& v) { os << std::get<N>(v) << ", "; tp<T, N + 1, L>::print(os, v); } };
    template<typename T, unsigned N> struct tp<T, N, N> { static void print(std::ostream& os, const T& v) { os << std::get<N>(v); } };
}
template<typename... Ts> std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) { os << '[';  aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::print(os, t); return os << ']'; }
template <class T, class = typename T::iterator, std::enable_if_t<!std::is_same<T, std::string>::value, int> = 0> std::ostream& operator<<(std::ostream& os, T const& a);
template <class T, class S> std::ostream& operator<<(std::ostream& os, std::pair<T, S> const& p) { return os << '[' << p.first << ", " << p.second << ']'; }
template <class T, class S> std::istream& operator>>(std::istream& is, std::pair<T, S>& p) { return is >> p.first >> p.second; }
template <class T, class, std::enable_if_t<!std::is_same<T, std::string>::value, int>>
std::ostream& operator<<(std::ostream& os, T const& a) { bool f = true; for (auto const& x : a) { os << (f ? "[" : ", ") << x; f = false; } return os << ']'; }
template <class T, size_t N, std::enable_if_t<!std::is_same<T, char>::value, int> = 0>
std::ostream& operator<<(std::ostream& os, const T(&a)[N]) { bool f = true; for (auto const& x : a) { os << (f ? "[" : ", ") << x; f = false; } return os << ']'; }
template <class T, class = decltype(std::begin(std::declval<T&>())), class = typename std::enable_if<!std::is_same<T, std::string>::value>::type>
std::istream& operator>>(std::istream& is, T& a) { for (auto& x : a) is >> x; return is; }
struct IOSetup { IOSetup(bool f) { if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); } std::cout << std::fixed << std::setprecision(15); } };
/* format */
template<typename... Ts> std::string format(const std::string& f, Ts... t) { size_t l = std::snprintf(nullptr, 0, f.c_str(), t...); std::vector<char> b(l + 1); std::snprintf(&b[0], l + 1, f.c_str(), t...); return std::string(&b[0], &b[0] + l); }
/* debug */
#define ENABLE_DEBUG
#ifdef ENABLE_DEBUG
#define DEBUGOUT std::cerr
#define debug(...) do{DEBUGOUT<<"  ";DEBUGOUT<<#__VA_ARGS__<<" :[DEBUG - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DEBUGOUT<<"    ";debug_func(__VA_ARGS__);}while(0);
inline void debug_func() { DEBUGOUT << std::endl; }
template <class Head, class... Tail> void debug_func(Head&& head, Tail&&... tail) { DEBUGOUT << head; if (sizeof...(Tail) == 0) { DEBUGOUT << " "; } else { DEBUGOUT << ", "; } debug_func(std::move(tail)...); }
#else
#define debug(...) void(0);
#endif
/* timer */
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    inline static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 3.0e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 3.0e9;
#endif
    }
    inline void reset() { t = time(); }
    inline void pause() { tmp = time(); }
    inline void restart() { paused += time() - tmp; }
    inline double elapsed_ms() { return (time() - t - paused) * 1000.0; }
};
/* fill */
template<typename A, size_t N, typename T>
void Fill(A(&array)[N], const T& val) {
    std::fill((T*)array, (T*)(array + N), val);
}
/* rand */
struct Xorshift {
    uint64_t x = 88172645463325252LL;
    Xorshift() {}
    Xorshift(unsigned seed) { set_seed(seed); }
    inline void set_seed(unsigned seed, int rep = 100) { x = uint64_t((seed + 1) * 10007); for (int i = 0; i < rep; i++) next_int(); }
    inline unsigned next_int() { x = x ^ (x << 7); return unsigned(x = x ^ (x >> 9)); }
    inline unsigned next_int(unsigned mod) { x = x ^ (x << 7); x = x ^ (x >> 9); return unsigned(x % mod); }
    inline unsigned next_int(unsigned l, unsigned r) { x = x ^ (x << 7); x = x ^ (x >> 9); return unsigned(x % (r - l + 1) + l); } // inclusive
    inline double next_double() { return double(next_int()) / UINT_MAX; }
};
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
inline std::vector<std::string> split(std::string str, const std::string& delim) {
    for (char& c : str) if (delim.find(c) != std::string::npos) c = ' ';
    std::istringstream iss(str);
    std::vector<std::string> parsed;
    std::string buf;
    while (iss >> buf) parsed.push_back(buf);
    return parsed;
}

Timer timer;
IOSetup iosetup(true);
Xorshift rnd;



struct P {
    int x, y;
    constexpr P(int x = 0, int y = 0) : x(x), y(y) {}
    P& operator+=(const P& p) { x += p.x; y += p.y; return *this; }
    P& operator-=(const P& p) { x -= p.x; y -= p.y; return *this; }
    P& operator-() { x = -x; y = -y; return *this; }
    bool operator<(const P& p) { return x == p.x ? y < p.y : x < p.x; }
    std::string str() const { return "[" + std::to_string(x) + ", " + std::to_string(y) + ']'; }
    friend std::ostream& operator<<(std::ostream& o, const P& p) { o << p.str(); return o; }
};
P operator+(const P& p1, const P& p2) { return P(p1) += p2; }
P operator-(const P& p1, const P& p2) { return P(p1) -= p2; }

constexpr P dir[] = {{0,1},{-1,0},{0,-1},{1,0}};

using Path = std::vector<P>;
using Board = std::vector<std::vector<int>>;

struct State {

    int N;
    Board board;
    std::vector<Path> paths;

    State() {}
    State(std::istream& in) {
        std::vector<int> a;
        std::string buf;
        while (in >> buf) {
            a.push_back(stoi(buf));
        }
        for (N = 1;; N++) if (N * N == a.size()) break;
        board.resize(N, std::vector<int>(N));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                board[i][j] = a[i * N + j];
            }
        }
    }

    inline bool is_inside(int x, int y) const { return 0 <= x && x < N && 0 <= y && y < N; }
    inline bool is_inside(const P& p) const { return is_inside(p.y, p.x); }

    void build_path(P p, const std::vector<P>& dir) {
        Path path({p});
        board[p.y][p.x]--;
        while(true) {
            bool ok = false;
            for(const auto& dp : dir) {
                auto np = p + dp;
                if(!is_inside(np) || board[p.y][p.x] != board[np.y][np.x] || board[p.y][p.x] == 0) continue;
                ok = true;
                board[np.y][np.x]--;
                path.push_back(np);
                p = np;
                break;
            }
            if(!ok) break;
        }
        paths.push_back(path);
    }

    P choose_top_left_highest_pos() const {
        int bmax = -1, ymax = -1, xmax = -1;
        for(int y = 0; y < N; y++) {
            for(int x = 0; x < N; x++) {
                if(bmax < board[y][x]) {
                    bmax = board[y][x];
                    ymax = y; xmax = x;
                }
            }
        }
        return P(xmax, ymax);
    }

    void solve(std::vector<P>& dir) {
        while(true) {
            auto p = choose_top_left_highest_pos();
            if(board[p.y][p.x] == 0) break;
            build_path(p, dir);
        }
    }

    void output(std::ostream& o) const {
        for(const auto& path : paths) {
            for(const auto& [x, y] : path) {
                o << y + 1 << ' ' << x + 1 << '\n';
            }
        }
    }
};

int main() {

    State init_state(std::cin);
    int best_score = 0;
    State best_state(init_state);

    std::vector<P> dir({{0,1},{-1,0},{0,-1},{1,0}});
    std::sort(dir.begin(), dir.end());

    do {
        State state(init_state);
        state.solve(dir);
        int score = 100000 - (int)state.paths.size();
        if(best_score < score) {
            best_state = state;
            best_score = score;
        }
    } while(std::next_permutation(dir.begin(), dir.end()));

    std::cerr << best_score << std::endl;
    best_state.output(std::cout);

    return 0;
}