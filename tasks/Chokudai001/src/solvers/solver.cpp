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

int N;
std::vector<std::vector<int>> A;

void load(std::istream& in) {
    std::vector<int> a;
    std::string buf;
    while (in >> buf) {
        a.push_back(stoi(buf));
    }
    for (N = 1;; N++) if (N * N == a.size()) break;
    A.resize(N, std::vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = a[i * N + j];
        }
    }
}

constexpr int di[] = {0, -1, 0, 1};
constexpr int dj[] = {1, 0, -1, 0};

inline bool is_inside(int i, int j) {
    return 0 <= i && i < N && 0 <= j && j < N;
}

using Path = std::vector<std::pair<int, int>>;

Path build_path(int i, int j) {
    Path ret({{i, j}});
    A[i][j]--;
    while(true) {
        bool ok = false;
        for(int d = 0; d < 4; d++) {
            int ni = i + di[d], nj = j + dj[d];
            if(!is_inside(ni, nj) || A[i][j] != A[ni][nj] || A[i][j] == 0) continue;
            ok = true;
            A[ni][nj]--;
            ret.emplace_back(ni, nj);
            i = ni; j = nj;
            break;
        }
        if(!ok) break;
    }
    return ret;
}

int main() {

    load(std::cin);

    Path ans;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            while(A[i][j]) {
                for(const auto& path : build_path(i, j)) {
                    ans.push_back(path);
                }
            }
        }
    }

    for(const auto& [i, j] : ans) {
        std::cout << i + 1 << ' ' << j + 1 << '\n';
    }

    return 0;
}