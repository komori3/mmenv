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



constexpr char TREE = 'T';
constexpr char BOX = 'b';
constexpr char ELF = 'e';
constexpr char ELF_PRESENT = 'E';
constexpr char ELF_BOX = 'B';
constexpr char PRESENT = 'P';
constexpr char EMPTY = '.';

constexpr int PRESENT_VALUE = 100;
constexpr int dr[] = { 0, -1, 0, 1 };
constexpr int dc[] = { 1, 0, -1, 0 };

struct TestCase;
using TestCasePtr = std::shared_ptr<TestCase>;
struct TestCase {
    int grid_size;
    int box_cost;
    double elf_spawn_prob;
    int initial_money;
    char initial_grid[32][32];
    TestCase(std::istream& in) {
        in >> grid_size >> box_cost >> elf_spawn_prob >> initial_money;
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                in >> initial_grid[y][x];
            }
        }
    }
};

struct Point {
    int x, y;
    Point(int x = 0, int y = 0) : x(x), y(y) {}
    std::string stringify() const {
        return format("Point [x=%d, y=%d]", x, y);
    }
};

std::vector<Point> create_rhombus(int cx, int cy, int dist) {
    std::vector<Point> poly;
    int x = cx + dist, y = cy, dx = -1, dy = -1;
    for (int i = 0; i < dist; i++) {
        poly.emplace_back(x, y);
        x += dx; y += dy;
    }
    dy = 1;
    for (int i = 0; i < dist; i++) {
        poly.emplace_back(x, y);
        x += dx; y += dy;
    }
    dx = 1;
    for (int i = 0; i < dist; i++) {
        poly.emplace_back(x, y);
        x += dx; y += dy;
    }
    dy = -1;
    for (int i = 0; i < dist; i++) {
        poly.emplace_back(x, y);
        x += dx; y += dy;
    }
    return poly;
}

std::vector<Point> create_inner_rhombus(int cx, int cy, int dist) {
    std::vector<Point> points({ {cx, cy} });
    for (int d = 1; d < dist; d++) {
        for (const auto& p : create_rhombus(cx, cy, d)) {
            points.push_back(p);
        }
    }
    return points;
}

std::vector<Point> create_rect(int x, int y, int w, int h) {
    std::vector<Point> points;
    for (int c = x; c < x + w - 1; c++) points.emplace_back(c, y);
    for (int r = y; r < y + h - 1; r++) points.emplace_back(x + w - 1, r);
    for (int c = x + w - 1; c > x; c--) points.emplace_back(c, y + h - 1);
    for (int r = y + h - 1; r > y; r--) points.emplace_back(x, r);
    return points;
}

std::vector<Point> create_inner_rect(int x, int y, int w, int h) {
    std::vector<Point> points;
    for (int r = y + 1; r < y + h - 1; r++) {
        for (int c = x + 1; c < x + w - 1; c++) {
            points.emplace_back(c, r);
        }
    }
    return points;
}

struct Simulator {

    Xorshift rnd;

    int N;
    int C;
    double spawn_prob;
    std::vector<int> spawn_idx;
    std::vector<Point> spawn_pos;

    char grid[32][32];
    int num_turns;
    int num_presents;
    int score;
    int money;

    void set_seed(int seed) { rnd.set_seed(seed); }

    void generate(TestCasePtr tc) {

        num_turns = 0;

        N = tc->grid_size;
        C = tc->box_cost;
        spawn_prob = tc->elf_spawn_prob;
        money = tc->initial_money;

        spawn_idx.resize(N * N + 1, -1);
        for (int i = 1; i <= N * N; i++) {
            double a = rnd.next_double();
            if (a < spawn_prob) {
                int p = rnd.next_int(N * 4 - 4);
                spawn_idx[i] = p;
            }
            else {
                spawn_idx[i] = -1;
            }
        }

        for (int x = 0; x < N - 1; x++) spawn_pos.emplace_back(x, 0);
        for (int y = 0; y < N - 1; y++) spawn_pos.emplace_back(N - 1, y);
        for (int x = N - 1; x > 0; x--) spawn_pos.emplace_back(x, N - 1);
        for (int y = N - 1; y > 0; y--) spawn_pos.emplace_back(0, y);

        num_presents = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                grid[i][j] = tc->initial_grid[i][j];
                if (grid[i][j] == PRESENT) num_presents++;
            }
        }

    }

    void move_elf(int r, int c, int d) {
        char e = grid[r][c];
        grid[r][c] = EMPTY;
        r += dr[d];
        c += dc[d];
        if (r < 0 || r >= N || c < 0 || c >= N) {
            if (e == ELF_PRESENT) num_presents--;
            return;
        }
        if (grid[r][c] == BOX) {
            grid[r][c] = ELF_BOX;
        }
        else if (grid[r][c] == PRESENT) {
            grid[r][c] = ELF_PRESENT;
        }
        else {
            grid[r][c] = e;
        }
    }

    void find_move(int y, int x) {
        // ç≈èâÇ…êiÇÒÇæï˚å¸ÇæÇØÉÅÉÇÇµÇƒ BFS
        std::queue<int> qd, qr, qc;
        qd.push(-1);
        qr.push(y);
        qc.push(x);
        char e = grid[y][x];
        std::vector<int> order({ 0, 1, 2, 3 });
        std::vector<int> visited(N * N, 0);
        int dirToBox = -1;
        while (!qd.empty()) {
            int d = qd.front(); qd.pop();
            int r = qr.front(); qr.pop();
            int c = qc.front(); qc.pop();
            if (e == ELF && grid[r][c] == PRESENT) {
                assert(d != -1);
                move_elf(y, x, d);
                return;
            }
            if (!visited[r * N + c]) {
                visited[r * N + c] = 1;
                shuffle_vector(order, rnd);
                for (int dd = 0; dd < 4; dd++) {
                    int dir = order[dd];
                    int nr = r + dr[dir];
                    int nc = c + dc[dir];
                    if ((nc >= 0 && nc < N && nr >= 0 && nr < N) || e == ELF_PRESENT || e == ELF_BOX) {
                        bool canMove = false;
                        if (e == ELF) {
                            if (grid[nr][nc] == EMPTY || grid[nr][nc] == PRESENT) canMove = true;
                        }
                        else {
                            if (nc < 0 || nc >= N || nr < 0 || nr >= N) {
                                move_elf(y, x, d == -1 ? dir : d);
                                return;
                            }
                            if (grid[nr][nc] == EMPTY) canMove = true;
                        }
                        if (grid[nr][nc] == BOX && dirToBox == -1) {
                            dirToBox = (d == -1 ? dir : d);
                        }
                        if (canMove) {
                            if (d == -1) qd.push(dir);
                            else qd.push(d);
                            qr.push(nr);
                            qc.push(nc);
                        }
                    }
                }
            }
        }
        if (dirToBox != -1 && e == ELF) {
            move_elf(y, x, dirToBox);
        }
    }

    bool simulate(const std::vector<Point>& box_points) {
        num_turns++;

        for (const auto [col, row] : box_points) {
            assert(!(col <= 0 || col >= N - 1 || row <= 0 || row >= N - 1));
            assert(grid[row][col] == EMPTY);
            assert(money >= C);
            grid[row][col] = BOX;
            money -= C;
        }

        std::vector<Point> elves;
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (grid[r][c] == ELF || grid[r][c] == ELF_PRESENT || grid[r][c] == ELF_BOX) {
                    elves.emplace_back(c, r);
                }
            }
        }

        for (const auto& pos : elves) {
            find_move(pos.y, pos.x);
        }

        int p = spawn_idx[num_turns];
        if (p >= 0) {
            int check = 0;
            while (true) {
                check++;
                if (check > N * 4) break;
                auto [pc, pr] = spawn_pos[p];
                if (grid[pr][pc] == EMPTY) {
                    grid[pr][pc] = ELF;
                    break;
                }
                p = (1 + p) % (N * 4 - 4);
            }
        }

        money++;

        score = PRESENT_VALUE * num_presents + money;

        return !(num_turns == N * N || num_presents == 0);
    }

    void print(std::ostream& out) const {
        std::ostringstream oss;
        oss << num_turns << std::endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                oss << grid[i][j];
            }
            oss << std::endl;
        }
        out << oss.str();
    }

};

struct State {

    int grid_size;
    int box_cost;
    double elf_spawn_prob;

    int turn;
    int money;
    char grid[32][32];

    void init(TestCasePtr tc) {
        grid_size = tc->grid_size;
        box_cost = tc->box_cost;
        elf_spawn_prob = tc->elf_spawn_prob;

        turn = 0;
        money = tc->initial_money;
        std::memcpy(grid, tc->initial_grid, sizeof(char) * 32 * 32);
    }

    void load(std::istream& in) {
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                in >> grid[y][x];
            }
        }
    }

};

int query(State& state, std::istream& in, std::ostream& out, const std::vector<Point>& box_points) {
    if (box_points.empty()) {
        out << -1;
    }
    else {
        for (const auto [x, y] : box_points) {
            out << y << ' ' << x << ' ';
        }
    }
    out << std::endl;
    int elapsed_ms;
    in >> elapsed_ms >> state.money;
    state.load(in);
    state.turn++;
    return elapsed_ms;
}

int simulate(TestCasePtr tc, const std::vector<Point>& box_points, int seed) {
    int box_cost = tc->box_cost;
    Simulator sim;
    sim.set_seed(seed);
    sim.generate(tc);
    while (true) {
        int money = sim.money;
        std::vector<Point> query;
        for (const auto [x, y] : box_points) {
            if (money >= box_cost && sim.grid[y][x] == '.') {
                query.emplace_back(x, y);
                money -= box_cost;
            }
        }
        bool res = sim.simulate(query);
        if (!res) break;
    }
    return sim.score;
}

double multi_simulate(TestCasePtr tc, const std::vector<Point>& box_points, int num_trial) {
    int score_sum = 0;
    for (int seed = 0; seed < num_trial; seed++) {
        int score = simulate(tc, box_points, seed);
        score_sum += score;
    }
    return (double)score_sum / num_trial;
}

struct LayoutEditor {

    int N;
    char grid[32][32];
    std::vector<Point> contour;
    bool contour_map[32][32];
    int cost;

    LayoutEditor(TestCasePtr tc) {
        N = tc->grid_size;
        std::memcpy(grid, tc->initial_grid, sizeof(char) * 32 * 32);
        std::memset(contour_map, 0, sizeof(bool) * 32 * 32);
        contour = create_rect(2, 2, N - 4, N - 4);
        for (const auto [x, y] : contour) contour_map[y][x] = true;
        cost = 0;
        for (const auto [x, y] : contour) {
            if (grid[y][x] == EMPTY) cost++;
        }
        // guard
        for (const auto [x, y] : create_rect(0, 0, N, N)) {
            grid[y][x] = PRESENT;
        }
    }

    bool is_corner(int r, int c) const {
        if (contour_map[r + dr[0]][c + dc[0]] && contour_map[r + dr[1]][c + dc[1]]) return true;
        if (contour_map[r + dr[1]][c + dc[1]] && contour_map[r + dr[2]][c + dc[2]]) return true;
        if (contour_map[r + dr[2]][c + dc[2]] && contour_map[r + dr[3]][c + dc[3]]) return true;
        if (contour_map[r + dr[3]][c + dc[3]] && contour_map[r + dr[0]][c + dc[0]]) return true;
        return false;
    }

    bool is_corner(const Point& p) {
        return is_corner(p.y, p.x);
    }

    int calc_connect_num_8(int y, int x) const {
        static constexpr int dr8[] = { 0, -1, -1, -1, 0, 1, 1, 1, 0 };
        static constexpr int dc8[] = { 1, 1, 0, -1, -1, -1, 0, 1, 1 };
        int n = 0;
        for (int k = 0; k < 8; k++) n += (int)(contour_map[y + dr8[k]][x + dc8[k]] != contour_map[y + dr8[k + 1]][x + dc8[k + 1]]);
        return n / 2;
    }

    int find(int y, int x) const {
        for (int i = 0; i < contour.size(); i++) {
            if (contour[i].x == x && contour[i].y == y) return i;
        }
        return -1;
    }

    void erase_corner() {
        int n = contour.size();
        for (int i = n - 1; i >= 0; i--) {
            auto [x, y] = contour[i];
            if (is_corner(y, x)) {
                contour_map[y][x] = false;
                std::swap(contour.back(), contour[i]);
                contour.pop_back();
                if (grid[y][x] == EMPTY) cost--;
            }
        }
    }

    std::pair<bool, int> can_move(int y, int x, int d) {
        static constexpr int dr8[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
        static constexpr int dc8[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
        if (grid[y + dr[d]][x + dc[d]] == PRESENT || contour_map[y + dr[d]][x + dc[d]]) return {false, -1};
        contour_map[y][x] = false;
        contour_map[y + dr[d]][x + dc[d]] = true;
        bool ok = true;
        for (int k = 0; k < 8; k++) {
            int ny = y + dr8[k], nx = x + dc8[k];
            if (contour_map[ny][nx] && calc_connect_num_8(ny, nx) != 2) ok = false;
        }
        if (calc_connect_num_8(y + dr[d], x + dc[d]) != 2) ok = false;
        contour_map[y + dr[d]][x + dc[d]] = false;
        contour_map[y][x] = true;
        return { ok, (int)(grid[y][x] == TREE) - (int)(grid[y + dr[d]][x + dc[d]] == TREE) };
    }

    void do_move(int y, int x, int d, int diff) {
        int idx = find(y, x);
        std::swap(contour[idx], contour.back());
        contour.pop_back();
        contour_map[y][x] = false;
        contour.emplace_back(x + dc[d], y + dr[d]);
        contour_map[y + dr[d]][x + dc[d]] = true;
        cost += diff;
    }

    bool check() const {
        int n = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (contour_map[i][j]) n++;
            }
        }
        if (n != contour.size()) return false;
        for (const auto& [x, y] : contour) {
            if (!contour_map[y][x]) return false;
        }
        return true;
    }

    void run() {
        erase_corner();
        dump(cost);
        for (int loop = 0; loop < 1000000; loop++) {
            int i = rnd.next_int(contour.size());
            auto [x, y] = contour[i];
            int d = rnd.next_int(4);
            auto [ok, diff] = can_move(y, x, d);
            if (!ok) continue;
            if (diff > 0) continue;
            do_move(y, x, d, diff);
            erase_corner();
        }
    }

#ifdef _MSC_VER

    void show(int delay = 0) const {
        cv::imshow("img", get_img());
        cv::waitKey(delay);
    }

    cv::Mat_<cv::Vec3b> get_img() const {
        static constexpr int grid_size = 32;
        cv::Mat_<cv::Vec3b> img(N * grid_size, N * grid_size, cv::Vec3b(255, 255, 255));
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                cv::Scalar color(255, 255, 255);
                if (grid[y][x] == TREE) {
                    color = cv::Scalar(0, 255, 0);
                }
                else if (grid[y][x] == PRESENT) {
                    color = cv::Scalar(0, 0, 255);
                }
                cv::Rect rect(grid_size * x, grid_size * y, grid_size, grid_size);
                cv::rectangle(img, rect, color, cv::FILLED);
            }
        }
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                cv::Rect rect(grid_size * x, grid_size * y, grid_size, grid_size);
                if (contour_map[y][x]) {
                    if (is_corner(y, x)) {
                        cv::rectangle(img, rect, cv::Scalar(255, 0, 255), 2);
                    }
                    else {
                        cv::rectangle(img, rect, cv::Scalar(255, 0, 0), 2);
                    }
                }
            }
        }
        return img;
    }
#endif
};

//#define LOCAL_TEST

int main() {

#ifdef LOCAL_TEST
    std::ifstream ifs("C:\\dev\\heuristic\\tasks\\MM131\\in\\2.in");
    std::istream& in = ifs;
#else
    std::istream& in = std::cin;
#endif

    TestCasePtr tc = std::make_shared<TestCase>(in);

    LayoutEditor layout(tc);
    layout.run();

    int grid_size = tc->grid_size;
    int box_cost = tc->box_cost;

    std::vector<Point> critical_points;
    int num_trial = 10;
    double best_score = -1.0;
    {
        // do nothing
        double score = multi_simulate(tc, {}, num_trial);
        dump("nothing", score);
        if (chmax(best_score, score)) {
            critical_points = {};
        }
    }
    {
        // rhombus
        int dist = (grid_size - 3) / 2;
        for (int y = 1 + dist; y + dist < grid_size - 1; y++) {
            for (int x = 1 + dist; x + dist < grid_size - 1; x++) {
                auto contour = create_rhombus(x, y, dist);
                double score = multi_simulate(tc, contour, num_trial);
                dump("rhombus", score);
                if (chmax(best_score, score)) {
                    critical_points = contour;
                }
            }
        }
    }
    {
        // rect
        auto contour = create_rect(2, 2, grid_size - 4, grid_size - 4);
        double score = multi_simulate(tc, contour, num_trial);
        dump("rect", score);
        if (chmax(best_score, score)) {
            critical_points = contour;
        }
    }
    {
        // low energy
        auto contour = layout.contour;
        double score = multi_simulate(tc, contour, num_trial);
        dump("low energy", score);
        if (chmax(best_score, score)) {
            critical_points = contour;
        }
    }


    State state;
    state.init(tc);

#ifdef LOCAL_TEST
    Simulator simulator;
    simulator.set_seed(1000000007);
    simulator.generate(tc);

    while (true) {
        std::vector<Point> box_points;
        int money = simulator.money;
        for (const auto [x, y] : critical_points) {
            if (money >= box_cost && simulator.grid[y][x] == '.') {
                box_points.emplace_back(x, y);
                money -= box_cost;
            }
        }
        bool res = simulator.simulate(box_points);
        //dump(tester.score);
        //tester.print(std::cerr);
        if (!res) break;
    }
    dump(simulator.score);
#else
    while (state.turn < grid_size * grid_size) {
        std::vector<Point> box_points;
        int money = state.money;
        for (const auto [x, y] : critical_points) {
            if (money >= box_cost && state.grid[y][x] == '.') {
                box_points.emplace_back(x, y);
                money -= box_cost;
            }
        }
        query(state, std::cin, std::cout, box_points);
    }
#endif

    return 0;
}