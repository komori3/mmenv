#include <bits/stdc++.h>
#include <random>
#ifdef _MSC_VER
#include <ppl.h>
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
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



int N, C;
double elfP;

struct Point {
    int x, y;
    Point(int x = 0, int y = 0) : x(x), y(y) {}
    std::string stringify() const {
        return format("Point [x=%d, y=%d]", x, y);
    }
};

struct Rect {
    int x, y, w, h;
    Rect(int x = 0, int y = 0, int w = 0, int h = 0) : x(x), y(y), w(w), h(h) {}
    std::string stringify() const {
        return format("Rect [x=%d, y=%d, w=%d, h=%d]", x, y, w, h);
    }
    Rect get_outer_shell() const { return Rect(x - 1, y - 1, w + 2, h + 2); }
    Rect get_inner_shell() const { return Rect(x + 1, y + 1, w - 2, h - 2); }
    std::vector<Point> get_boundary_points() const {
        std::vector<Point> points;
        for (int j = 0; j < w - 1; j++) points.emplace_back(x + j, y);
        for (int i = 0; i < h - 1; i++) points.emplace_back(x + w - 1, y + i);
        for (int j = w - 1; j > 0; j--) points.emplace_back(x + j, y + h - 1);
        for (int i = h - 1; i > 0; i--) points.emplace_back(x, y + i);
        return points;
    }
};

struct State {
    int money;
    char grid[32][32];
    State() {}

    void load(std::istream& in) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                in >> grid[y][x];
            }
        }
    }

    int count(const Rect& roi, char type) const {
        int res = 0;
        for (int y = roi.y; y < roi.y + roi.w; y++) {
            for (int x = roi.x; x < roi.x + roi.w; x++) {
                res += (grid[y][x] == type);
            }
        }
        return res;
    }

};

int main() {

    State state;
    std::cin >> N >> C >> elfP >> state.money;

    state.load(std::cin);

    Rect best_roi;
    {
        int max_presents = -1;
        for (int y = 1; y + 4 < N; y++) {
            for (int x = 1; x + 4 < N; x++) {
                Rect roi(x + 1, y + 1, 3, 3);
                dump(roi, state.count(roi, 'P'));
                int num_presents = state.count(roi, 'P');
                if (chmax(max_presents, num_presents)) {
                    best_roi = roi;
                    max_presents = num_presents;
                }
            }
        }
        dump(best_roi, max_presents);
        dump(best_roi.get_outer_shell().get_boundary_points());
    }

    auto critical_points = best_roi.get_outer_shell().get_boundary_points();

    for (int turn = 0; turn < N * N; turn++) {
        bool updated = false;
        for (const auto [x, y] : critical_points) {
            if (state.money >= C && state.grid[y][x] == '.') {
                std::cout << y << ' ' << x << ' ';
                state.money -= C;
                updated = true;
            }
        }
        if (!updated) std::cout << "-1";
        std::cout << std::endl;
        int elapsedTime;
        //read elapsed time
        std::cin >> elapsedTime;
        //read the money
        std::cin >> state.money;
        //read the updated grid
        state.load(std::cin);
    }

}