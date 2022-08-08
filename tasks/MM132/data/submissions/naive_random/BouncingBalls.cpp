#include <bits/stdc++.h>
#include <random>
#ifdef _MSC_VER
#include <ppl.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
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


using pii = std::pair<int, int>;
constexpr int dr[] = { 0, -1, 0, 1 };
constexpr int dc[] = { 1, 0, -1, 0 };



namespace NInput {

    int N;
    int B;
    int bonus_val;
    std::vector<std::string> grid;

    void init(std::istream& in) {
        in >> N >> B >> bonus_val;
        grid.resize(N + 2, std::string(N + 2, '#')); // #: outside
        for (int r = 1; r <= N; r++) {
            for (int c = 1; c <= N; c++) {
                in >> grid[r][c];
            }
        }
        dump(grid);
    }

}

void print(const std::vector<pii>& guns, std::vector<std::string> grid) {
    for (const auto& [r, c] : guns) grid[r][c] = 'b';
    for (const auto& v : grid) std::cerr << v << '\n';
    std::cerr << std::endl;
}

int simulate(std::vector<pii> guns, std::vector<std::string> grid) {
    
    const int N = NInput::N;
    const int B = NInput::B;
    const int bonus_val = NInput::bonus_val;

    //assert(guns.size() == B);
    //assert(grid.size() == N + 2);
    for (int r = 0; r < N + 2; r++) {
        //assert(grid[r].size() == N + 2);
        for (int c = 0; c < N + 2; c++) {
            if (NInput::grid[r][c] != '.') {
                //assert(NInput::grid[r][c] == grid[r][c]);
            }
        }
    }

    auto get_dir = [&](const pii& p) {
        if (p.second == 0) return 0;
        if (p.first == N + 1) return 1;
        if (p.second == N + 1) return 2;
        if (p.first == 0) return 3;
        //assert(false);
        return -1;
    };

    std::vector<int> dirs(B);
    for (int i = 0; i < B; i++) {
        dirs[i] = get_dir(guns[i]);
    }

    int score = 0;
    std::vector<std::vector<int>> num_balls(N + 2, std::vector<int>(N + 2, 0));
    for (const auto& [r, c] : guns) num_balls[r][c]++;

    while (true) {
        bool end_flag = false;
        // ’e‚ÌˆÚ“®
        for (int i = 0; i < B; i++) {
            auto& [r, c] = guns[i];

            num_balls[r][c]--;
            r += dr[dirs[i]]; c += dc[dirs[i]];
            num_balls[r][c]++;

            if (grid[r][c] == '*') score += bonus_val;
            else if (grid[r][c] == '/' || grid[r][c] == '\\') score++;

            if (dirs[i] == 0) {
                if (grid[r][c] == '\\') dirs[i] = 3;
                else if (grid[r][c] == '/') dirs[i] = 1;
            }
            else if (dirs[i] == 1) {
                if (grid[r][c] == '\\') dirs[i] = 2;
                else if (grid[r][c] == '/') dirs[i] = 0;
            }
            else if (dirs[i] == 2) {
                if (grid[r][c] == '\\') dirs[i] = 1;
                else if (grid[r][c] == '/') dirs[i] = 3;
            }
            else {
                if (grid[r][c] == '\\') dirs[i] = 0;
                else if (grid[r][c] == '/') dirs[i] = 2;
            }

            if (grid[r][c] == '#') end_flag = true;
        }
        // ”½ŽË”Â‚Ì•Ï‰»
        for (const auto& [r, c] : guns) {
            if (num_balls[r][c] == 1 && NInput::grid[r][c] == '.' && grid[r][c] != '.') {
                //assert(grid[r][c] == '/' || grid[r][c] == '\\');
                if (grid[r][c] == '/') grid[r][c] = '\\';
                else grid[r][c] = '/';
            }
        }

        if (end_flag) break;
    }

    return score;
}

void output(std::ostream& out, const std::vector<pii>& guns, const std::vector<std::string>& grid) {
    for (const auto& [r, c] : guns) {
        out << r - 1 << ' ' << c - 1 << '\n';
    }
    for (int r = 1; r <= NInput::N; r++) {
        for (int c = 1; c <= NInput::N; c++) {
            out << grid[r][c] << '\n';
        }
    }
    out.flush();
}

std::pair<std::vector<pii>, std::vector<std::string>> generate_random(Xorshift& rnd) {
    int N = NInput::N;
    int B = NInput::B;

    std::set<pii> seen;
    std::vector<pii> guns;
    std::vector<std::string> grid(NInput::grid);

    // gun locations
    while((int)guns.size() < B) {
        int type = rnd.next_int(4);
        int loc = rnd.next_int(N) + 1;
        pii p;
        if (type == 0) p = pii(0, loc);
        else if (type == 1) p = pii(N + 1, loc);
        else if (type == 2) p = pii(loc, 0);
        else p = pii(loc, N + 1);
        if (seen.count(p)) continue;
        seen.insert(p);
        guns.push_back(p);
    }

    char cells[] = { '/','\\','.' };
    // grid
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++)
        {
            if (NInput::grid[r + 1][c + 1] == '.')
            {
                int type = rnd.next_int(2);
                grid[r + 1][c + 1] = cells[type];
            }
        }
    }

    return { guns, grid };
}

int main() {

#if 0
    std::ifstream ifs("C:\\Users\\komori3\\OneDrive\\dev\\heuristic\\tasks\\MM132\\in\\1.in");
    std::istream& in = ifs;
    std::ofstream ofs("C:\\Users\\komori3\\OneDrive\\dev\\heuristic\\tasks\\MM132\\out\\1.out");
    std::ostream& out = ofs;
#else
    std::istream& in = std::cin;
    std::ostream& out = std::cout;
#endif

    NInput::init(in);

    auto [best_guns, best_grid] = generate_random(rnd);
    int best_score = simulate(best_guns, best_grid);
    dump(best_score);

    int loop = 0;
    while (timer.elapsed_ms() < 9000) {
        auto [guns, grid] = generate_random(rnd);
        int score = simulate(guns, grid);
        if (best_score < score) {
            best_score = score;
            best_grid = grid;
            best_guns = guns;
            dump(loop, best_score);
        }
        loop++;
    }

    output(out, best_guns, best_grid);

    return 0;
}