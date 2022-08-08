#define _CRT_NONSTDC_NO_WARNINGS
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#ifdef _MSC_VER
#define ENABLE_VIS
#define ENABLE_DUMP
#include <conio.h>
#include <ppl.h>
#include <filesystem>
#ifdef ENABLE_VIS
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#endif
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
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
} g_timer;

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

using std::vector, std::string;
using std::cin, std::cout, std::cerr, std::endl;
using ll = long long;
using pii = std::pair<int, int>;




#define BATCH_TEST
#ifdef BATCH_TEST
#undef ENABLE_DUMP
#endif

constexpr int TOP = 0;
constexpr int BOTTOM = 1;
constexpr int FRONT = 2;
constexpr int BACK = 3;
constexpr int LEFT = 4;
constexpr int RIGHT = 5;

struct Input;
using InputPtr = std::shared_ptr<Input>;
struct Input {
    int N;
    int V;
    double B;
    vector<vector<int>> grid;
    Input(std::istream& in) {
        in >> N >> V >> B;
        grid.resize(N, vector<int>(N));
        in >> grid;
    }
    string stringify() const {
        string res;
        res += format("N=%d, V=%d, B=%f\n", N, V, B);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                res += format("%d ", grid[i][j]);
            }
            res += '\n';
        }
        return res;
    }
};

struct Dice {
    int a[6];
    Dice() {}
    Dice(int t, int bo, int f, int ba, int l, int r)
        : a{ t, bo, f, ba, l, r } {}
    inline int& operator[] (int i) { return a[i]; }
    inline int operator[] (int i) const { return a[i]; }
    string stringify() const {
        string res;
        for (int x : a) res += std::to_string(x) + ' ';
        res.pop_back();
        return res;
    }
};

struct Solution {
    Dice dice;
    vector<pii> moves;
    void output(std::ostream& out) const {
        for (int i = 0; i < 6; i++) out << dice[i] << '\n';
        out << moves.size() << '\n';
        for (const auto [r, c] : moves) {
            out << r << ' ' << c << '\n';
        }
    }
};

double compute_score(const Input& input, const Solution& sol) {
    static constexpr int inf = -1000;

    const int N = input.N;
    const int V = input.V;
    const double B = input.B;
    const auto& grid = input.grid;

    int r = inf, c = inf;
    int pr = inf, pc = inf, sr = inf, sc = inf;
    double score = 0;
    Dice dice(sol.dice);

    int num_moves = sol.moves.size();

    auto make_move = [&]() {
        //right
        if (c == pc + 1)
        {
            int temp = dice[RIGHT];
            dice[RIGHT] = dice[TOP];
            dice[TOP] = dice[LEFT];
            dice[LEFT] = dice[BOTTOM];
            dice[BOTTOM] = temp;
        }
        //left
        else if (c == pc - 1)
        {
            int temp = dice[RIGHT];
            dice[RIGHT] = dice[BOTTOM];
            dice[BOTTOM] = dice[LEFT];
            dice[LEFT] = dice[TOP];
            dice[TOP] = temp;
        }
        //up
        else if (r == pr - 1)
        {
            int temp = dice[FRONT];
            dice[FRONT] = dice[BOTTOM];
            dice[BOTTOM] = dice[BACK];
            dice[BACK] = dice[TOP];
            dice[TOP] = temp;
        }
        //down
        else if (r == pr + 1)
        {
            int temp = dice[FRONT];
            dice[FRONT] = dice[TOP];
            dice[TOP] = dice[BACK];
            dice[BACK] = dice[BOTTOM];
            dice[BOTTOM] = temp;
        }

        if (dice[BOTTOM] == abs(grid[r][c])) {
            score += grid[r][c];
        }

        pr = r;
        pc = c;
    };

    for (int id = 0; id < num_moves; id++) {
        r = sol.moves[id].first;
        c = sol.moves[id].second;
        if (id == 0) {
            sr = r;
            sc = c;
        }
        make_move();
    }

    score *= (abs(r - sr) + abs(c - sc) == 1) ? B : 1.0;
    return score;
}

struct HamiltonianPathOn2DGrid {

    static constexpr int di[] = { 0, -1, 0, 1 };
    static constexpr int dj[] = { 1, 0, -1, 0 };

    Xorshift rnd;
    std::vector<int> dirs;

    int N;
    std::vector<std::pair<int, int>> points;

    HamiltonianPathOn2DGrid(int N, unsigned seed = 0) : N(N) {
        rnd.reseed(seed);
        dirs = { 0, 1, 2, 3 };
    }

    bool is_inside(int i, int j) const {
        return 0 <= i && i < N && 0 <= j && j < N;
    }

    int find(int i, int j) const {
        for (int k = 0; k < points.size(); k++) {
            if (i != points[k].first || j != points[k].second) continue;
            return k;
        }
        return -1;
    }

    void add_point(int i, int j) {
        points.emplace_back(i, j);
    }

    void initialize() {
        for (int i = 0; i < N; i++) {
            if (i % 2 == 0) {
                for (int j = 0; j < N; j++) {
                    add_point(i, j);
                }
            }
            else {
                for (int j = N - 1; j >= 0; j--) {
                    add_point(i, j);
                }
            }
        }
    }

    void move_reverse() {
        std::reverse(points.begin(), points.end());
    }

    int move_backbite() {
        shuffle_vector(dirs, rnd);
        auto [ei, ej] = points.back();
        for (int d : dirs) {
            int ni = ei + di[d], nj = ej + dj[d];
            if (!is_inside(ni, nj)) continue;
            int idx = find(ni, nj);
            std::reverse(points.begin(), points.end());
            std::reverse(points.begin(), points.end() - idx - 1);
            return idx;
        }
        return -1;
    }

    void undo_backbite(int idx) {
        std::reverse(points.begin(), points.end() - idx - 1);
        std::reverse(points.begin(), points.end());
    }

    bool is_loop() const {
        auto [i1, j1] = points.front();
        auto [i2, j2] = points.back();
        return abs(i1 - i2) + abs(j1 - j2) == 1;
    }

#ifdef HAVE_OPENCV_HIGHGUI
    void show(int delay = 0) const {
        int grid_size = 800 / N;
        int img_size = grid_size * N;

        auto to_img_coord = [&](int i, int j) {
            return cv::Point(grid_size * j + grid_size / 2, grid_size * i + grid_size / 2);
        };

        auto get_color = [](double ratio) {
            cv::Scalar color;
            if (ratio < 0.5) {
                // blue to purple
                int val = std::round(ratio * 255 / 0.5);
                color = cv::Scalar(255, 0, val);
            }
            else {
                int val = std::round((ratio - 0.5) * 255 / 0.5);
                color = cv::Scalar(255 - val, 0, 255);
            }
            return color;
        };

        //int line_width = std::max(1, grid_size / 4);
        int line_width = 2;
        cv::Mat_<cv::Vec3b> img(img_size, img_size, cv::Vec3b(255, 255, 255));
        for (int k = 1; k < points.size(); k++) {
            auto [i1, j1] = points[k - 1];
            auto [i2, j2] = points[k];
            cv::arrowedLine(img, to_img_coord(i1, j1), to_img_coord(i2, j2), get_color(double(k) / points.size()), line_width, 8, 0, 0.4);
        }

        cv::imshow("img", img);
        cv::waitKey(delay);
    }
#endif

};

double calc_path_score(const Input& input, const HamiltonianPathOn2DGrid& hgrid) {
    static constexpr int inf = -1000;

    const int N = input.N;
    const int V = input.V;
    const double B = input.B;
    const auto& grid = input.grid;

    int r = inf, c = inf;
    int pr = inf, pc = inf, sr = inf, sc = inf;
    Dice dice(0, 1, 2, 3, 4, 5);
    // i番目の数字をjに設定した時のgain
    auto gain = make_vector(0, 6, V + 1);

    int num_moves = N * N;

    auto make_move = [&]() {
        //right
        if (c == pc + 1)
        {
            int temp = dice[RIGHT];
            dice[RIGHT] = dice[TOP];
            dice[TOP] = dice[LEFT];
            dice[LEFT] = dice[BOTTOM];
            dice[BOTTOM] = temp;
        }
        //left
        else if (c == pc - 1)
        {
            int temp = dice[RIGHT];
            dice[RIGHT] = dice[BOTTOM];
            dice[BOTTOM] = dice[LEFT];
            dice[LEFT] = dice[TOP];
            dice[TOP] = temp;
        }
        //up
        else if (r == pr - 1)
        {
            int temp = dice[FRONT];
            dice[FRONT] = dice[BOTTOM];
            dice[BOTTOM] = dice[BACK];
            dice[BACK] = dice[TOP];
            dice[TOP] = temp;
        }
        //down
        else if (r == pr + 1)
        {
            int temp = dice[FRONT];
            dice[FRONT] = dice[TOP];
            dice[TOP] = dice[BACK];
            dice[BACK] = dice[BOTTOM];
            dice[BOTTOM] = temp;
        }

        gain[dice[BOTTOM]][abs(grid[r][c])] += grid[r][c];

        pr = r;
        pc = c;
    };

    for (int id = 0; id < num_moves; id++) {
        std::tie(r, c) = hgrid.points[id];
        if (id == 0) {
            sr = r;
            sc = c;
        }
        make_move();
    }

    double score = 0;
    for (int i = 0; i < 6; i++) {
        score += *std::max_element(gain[i].begin() + 1, gain[i].end());
    }

    {
        auto [tr, tc] = hgrid.points.back();
        score *= (abs(sr - tr) + abs(sc - tc) == 1) ? B : 1.0;
    }

    return score;
}

Dice calc_best_dice(const Input& input, const HamiltonianPathOn2DGrid& hgrid) {
    static constexpr int inf = -1000;

    const int N = input.N;
    const int V = input.V;
    const double B = input.B;
    const auto& grid = input.grid;

    int r = inf, c = inf;
    int pr = inf, pc = inf, sr = inf, sc = inf;
    Dice dice(0, 1, 2, 3, 4, 5);
    // i番目の数字をjに設定した時のgain
    auto gain = make_vector(0, 6, V + 1);

    int num_moves = N * N;

    auto make_move = [&]() {
        //right
        if (c == pc + 1)
        {
            int temp = dice[RIGHT];
            dice[RIGHT] = dice[TOP];
            dice[TOP] = dice[LEFT];
            dice[LEFT] = dice[BOTTOM];
            dice[BOTTOM] = temp;
        }
        //left
        else if (c == pc - 1)
        {
            int temp = dice[RIGHT];
            dice[RIGHT] = dice[BOTTOM];
            dice[BOTTOM] = dice[LEFT];
            dice[LEFT] = dice[TOP];
            dice[TOP] = temp;
        }
        //up
        else if (r == pr - 1)
        {
            int temp = dice[FRONT];
            dice[FRONT] = dice[BOTTOM];
            dice[BOTTOM] = dice[BACK];
            dice[BACK] = dice[TOP];
            dice[TOP] = temp;
        }
        //down
        else if (r == pr + 1)
        {
            int temp = dice[FRONT];
            dice[FRONT] = dice[TOP];
            dice[TOP] = dice[BACK];
            dice[BACK] = dice[BOTTOM];
            dice[BOTTOM] = temp;
        }

        gain[dice[BOTTOM]][abs(grid[r][c])] += grid[r][c];

        pr = r;
        pc = c;
    };

    for (int id = 0; id < num_moves; id++) {
        std::tie(r, c) = hgrid.points[id];
        if (id == 0) {
            sr = r;
            sc = c;
        }
        make_move();
    }

    Dice res;
    for (int i = 0; i < 6; i++) {
        res[i] = std::distance(gain[i].begin(), std::max_element(gain[i].begin() + 1, gain[i].end()));
    }

    return res;
}

void test() {

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    std::ifstream ifs("tester/in/2.in");
    std::ofstream ofs("tester/out/2.out");
    std::istream& in = ifs;
    std::ostream& out = ofs;

    Input input(in);
    int N = input.N;
    int V = input.V;

    Solution sol;
    sol.dice[TOP] = (0 % V) + 1;
    sol.dice[BOTTOM] = (1 % V) + 1;
    sol.dice[FRONT] = (2 % V) + 1;
    sol.dice[BACK] = (3 % V) + 1;
    sol.dice[LEFT] = (4 % V) + 1;
    sol.dice[RIGHT] = (5 % V) + 1;

    //move in a spiral
    int dr[] = { 0,1,0,-1 };
    int dc[] = { 1,0,-1,0 };
    int dir = 0;
    int r = 0;
    int c = 0;

    auto seen = make_vector(false, N, N);
    for (int i = 0; i < N * N; i++) {
        sol.moves.emplace_back(r, c);
        seen[r][c] = true;
        int r2 = r + dr[dir];
        int c2 = c + dc[dir];
        if (r2 < 0 || r2 >= N || c2 < 0 || c2 >= N || seen[r2][c2]) {
            dir = (dir + 1) % 4;
            r2 = r + dr[dir];
            c2 = c + dc[dir];
        }
        r = r2;
        c = c2;
    }

    int score = (int)round(compute_score(input, sol));
    dump(score);

    assert(score == 190);
}

int main(int argc, char** argv) {

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#if _MSC_VER
    std::ifstream ifs("tester/in/2.in");
    std::ofstream ofs("tester/out/2.out");
    std::istream& in = ifs;
    std::ostream& out = ofs;
#else
    std::istream& in = cin;
    std::ostream& out = cout;
#endif

    Input input(in);
    int N = input.N;
    int V = input.V;

    // hamiltonian path を求める
    // dice を転がす
    // 最適な出目は自明に求まる

    double max_score = -1;
    Solution best_sol;

    HamiltonianPathOn2DGrid grid(input.N);
    grid.initialize();
    int loop = 0;
    while (timer.elapsed_ms() < 9500) {
        loop++;
        double score = calc_path_score(input, grid);
        if (chmax(max_score, score)) {
            dump(loop, max_score);
            best_sol.dice = calc_best_dice(input, grid);
            best_sol.moves = grid.points;
        }
        grid.move_backbite();
    }

    dump(best_sol.dice);

    best_sol.output(out);

    return 0;
}