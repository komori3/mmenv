#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#ifdef _MSC_VER
#include <ppl.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif
/* const */
constexpr double PI = 3.141592653589793238462643;
/* io */
namespace aux {
    template<typename T, unsigned N, unsigned L> struct tp { static void print(std::ostream& os, const T& v) { os << std::get<N>(v) << ", "; tp<T, N + 1, L>::print(os, v); } };
    template<typename T, unsigned N> struct tp<T, N, N> { static void print(std::ostream& os, const T& v) { os << std::get<N>(v); } };
}
template<typename... Ts> std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) { os << '['; aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::print(os, t); os << ']'; return os; }
template <class T, class = typename T::iterator, std::enable_if_t<!std::is_same<T, std::string>::value, int> = 0> std::ostream& operator<<(std::ostream& os, T const& a);
template <class T, class S> std::ostream& operator<<(std::ostream& os, std::pair<T, S> const& p) { return os << '[' << p.first << ", " << p.second << ']'; }
template <class T, class S> std::istream& operator>>(std::istream& is, std::pair<T, S>& p) { return is >> p.first >> p.second; }
template <class T, class, std::enable_if_t<!std::is_same<T, std::string>::value, int>>
std::ostream& operator<<(std::ostream& os, T const& a) { bool f = true; os << '['; for (auto const& x : a) { os << (f ? "" : ", ") << x; f = false; } os << ']'; return os; }
template <class T, size_t N, std::enable_if_t<!std::is_same<T, char>::value, int> = 0>
std::ostream& operator<<(std::ostream& os, const T(&a)[N]) { bool f = true; os << '['; for (auto const& x : a) { os << (f ? "" : ", ") << x; f = false; } os << ']'; return os; }
template <class T, class = decltype(std::begin(std::declval<T&>())), class = typename std::enable_if<!std::is_same<T, std::string>::value>::type>
std::istream& operator>>(std::istream& is, T& a) { for (auto& x : a) is >> x; return is; }
struct IOSetup { IOSetup(bool f) { if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); } std::cout << std::fixed << std::setprecision(15); } } iosetup(true);
/* format */
template<typename... Ts> std::string format(const std::string& f, Ts... t) { size_t l = std::snprintf(nullptr, 0, f.c_str(), t...); std::vector<char> b(l + 1); std::snprintf(&b[0], l + 1, f.c_str(), t...); return std::string(&b[0], &b[0] + l); }
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
/* misc */
struct Timer {
    double t = 0, paused = 0, tmp; Timer() { reset(); } static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 3.0e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 3.0e9;
#endif
    } void reset() { t = time(); } void pause() { tmp = time(); } void restart() { paused += time() - tmp; } double elapsedMs() { return (time() - t - paused) * 1000.0; }
} timer;
struct Xorshift {
    uint64_t x = 88172645463325252LL;
    void set_seed(unsigned seed, int rep = 100) { x = (seed + 1) * 10007; for (int i = 0; i < rep; i++) next_int(); }
    unsigned next_int() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    unsigned next_int(unsigned mod) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % mod; }
    unsigned next_int(unsigned l, unsigned r) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % (r - l + 1) + l; }
    double next_double() { return double(next_int()) / UINT_MAX; }
} rnd;
template<typename T> void shuffle_vector(std::vector<T>& v, Xorshift& rnd) { int n = v.size(); for (int i = n - 1; i >= 1; i--) { int r = rnd.next_int(i); std::swap(v[i], v[r]); } }
std::vector<std::string> split(std::string str, const std::string& delim) { for (char& c : str) if (delim.find(c) != std::string::npos) c = ' '; std::istringstream iss(str); std::vector<std::string> parsed; std::string buf; while (iss >> buf) parsed.push_back(buf); return parsed; }

using ll = long long;
using pii = std::pair<int, int>;
using pll = std::pair<ll, ll>;
using pdd = std::pair<double, double>;



constexpr int BOARD_SIZE = 16;
constexpr int NUM_VEGES = 5000;
constexpr int MAX_TURN = 1000;

struct Vege;
using VegePtr = std::shared_ptr<Vege>;
struct Vege {
    int row, col;
    int begin, end;
    int value;
    Vege(int row = -1, int col = -1, int begin = -1, int end = -1, int value = -1)
        : row(row), col(col), begin(begin), end(end), value(value)
    {}

    std::string str() const {
        return format("Vege [row=%d, col=%d, begin=%d, end=%d, value=%d]", row, col, begin, end, value);
    }
    friend std::ostream& operator<<(std::ostream& o, const Vege& obj) {
        o << obj.str();
        return o;
    }
    friend std::ostream& operator<<(std::ostream& o, const VegePtr& obj) {
        o << obj->str();
        return o;
    }
};

struct State {
    using Action = std::vector<int>;

    std::vector<VegePtr> veges;
    std::vector<std::vector<VegePtr>> vege_begin;
    std::vector<std::vector<VegePtr>> vege_end;

    std::vector<std::vector<VegePtr>> vege_board;

    int num_machine;
    std::vector<std::vector<bool>> has_machine;

    int turn;
    int money;

    std::vector<Action> actions;
    std::vector<std::vector<VegePtr>> removed;
    std::vector<int> money_hist;

    State() {}
    State(const std::vector<VegePtr>& veges) :
        veges(veges),
        vege_begin(MAX_TURN + 1),
        vege_end(MAX_TURN + 1),
        vege_board(BOARD_SIZE + 2, std::vector<VegePtr>(BOARD_SIZE + 2, nullptr)),
        num_machine(0),
        has_machine(BOARD_SIZE + 2, std::vector<bool>(BOARD_SIZE + 2, false)),
        turn(0),
        money(1),
        removed(MAX_TURN),
        money_hist(MAX_TURN, -1)
    {
        for (const auto& vege : veges) {
            vege_begin[vege->begin].push_back(vege);
            vege_end[vege->end].push_back(vege);
        }
        appear(); // ‰Šú”z’u
    }

    void appear() {
        for (const auto& vege : vege_begin[turn]) {
            vege_board[vege->row][vege->col] = vege;
        }
    }

    inline int purchase_cost() const {
        return (num_machine + 1) * (num_machine + 1) * (num_machine + 1);
    }

    void purchase(int r, int c) {
        int cost = purchase_cost();
        assert(cost <= money);
        money -= cost;
        num_machine++;
        has_machine[r][c] = true;
        actions.push_back({ r, c });
    }

    void undo_purchase() {
        auto action = actions.back();
        actions.pop_back();
        int r = action[0], c = action[1];
        has_machine[r][c] = false;
        num_machine--;
        int cost = purchase_cost();
        money += cost;
    }

    void move(int r1, int c1, int r2, int c2) {
        assert(has_machine[r1][c1] && !has_machine[r2][c2]);
        has_machine[r1][c1] = false;
        has_machine[r2][c2] = true;
        actions.push_back({ r1, c1, r2, c2 });
    }

    void undo_move() {
        auto action = actions.back();
        actions.pop_back();
        int r1 = action[0], c1 = action[1], r2 = action[2], c2 = action[3];
        has_machine[r2][c2] = false;
        has_machine[r1][c1] = true;
    }

    void pass() {
        actions.push_back({ -1 });
    }

    void undo_pass() {
        actions.pop_back();
    }

    int calc_cc(int sr, int sc) const {
        static constexpr int dr[] = { 0, -1, 0, 1 };
        static constexpr int dc[] = { 1, 0, -1, 0 };
        static bool visited[BOARD_SIZE + 2][BOARD_SIZE + 2];
        memset(visited, 0, sizeof(bool) * (BOARD_SIZE + 2) * (BOARD_SIZE + 2));
        int cnt = 0;
        std::queue<pii> qu;
        qu.emplace(sr, sc);
        visited[sr][sc] = true;
        cnt++;
        while (!qu.empty()) {
            auto [r, c] = qu.front(); qu.pop();
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d], nc = c + dc[d];
                if (visited[nr][nc] || !has_machine[nr][nc]) continue;
                qu.emplace(nr, nc);
                visited[nr][nc] = true;
                cnt++;
            }
        }
        return cnt;
    }

    void simulate(const Action& action) {
        // action
        if (action.size() == 2) {
            purchase(action[0], action[1]);
        }
        else if (action.size() == 4) {
            move(action[0], action[1], action[2], action[3]);
        }
        else {
            pass();
        }

        money_hist[turn] = money;

        // calc score
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                if (vege_board[r][c] && has_machine[r][c]) {
                    money += vege_board[r][c]->value * calc_cc(r, c);
                    removed[turn].push_back(vege_board[r][c]);
                    vege_board[r][c] = nullptr;
                }
            }
        }
        // disappear
        for (const auto& vege : vege_end[turn]) {
            if (vege_board[vege->row][vege->col]) {
                removed[turn].push_back(vege);
                vege_board[vege->row][vege->col] = nullptr;
            }
        }
        turn++;
        // appear
        for (const auto& vege : vege_begin[turn]) {
            vege_board[vege->row][vege->col] = vege;
        }
    }

    void undo() {
        // appear
        for (const auto& vege : vege_begin[turn]) {
            assert(vege == vege_board[vege->row][vege->col]);
            vege_board[vege->row][vege->col] = nullptr;
        }
        turn--;
        // harvest & disappear
        for (const auto& vege : removed[turn]) {
            assert(!vege_board[vege->row][vege->col]);
            vege_board[vege->row][vege->col] = vege;
        }
        removed[turn].clear();

        money = money_hist[turn];
        money_hist[turn] = -1;

        const auto& action = actions.back();
        if (action.size() == 2) {
            undo_purchase();
        }
        else if (action.size() == 4) {
            undo_move();
        }
        else {
            undo_pass();
        }
    }

    Action best_purchase() {
        int best_r = -1, best_c = -1, best_value = -1;
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                if (has_machine[r][c]) continue;
                has_machine[r][c] = true;
                int value = (vege_board[r][c] ? vege_board[r][c]->value : 0) * calc_cc(r, c);
                has_machine[r][c] = false;
                if (best_value < value) {
                    best_value = value;
                    best_r = r;
                    best_c = c;
                }
            }
        }
        return { best_r, best_c };
    }

    Action best_move2() {
        Action best_action;
        int best_money = -1;
        for (int r1 = 1; r1 <= BOARD_SIZE; r1++) {
            for (int c1 = 1; c1 <= BOARD_SIZE; c1++) {
                if (!has_machine[r1][c1]) continue;
                for (int r2 = 1; r2 <= BOARD_SIZE; r2++) {
                    for (int c2 = 1; c2 <= BOARD_SIZE; c2++) {
                        if (has_machine[r2][c2]) continue;
                        simulate({ r1, c1, r2, c2 });
                        if (best_money < money) {
                            best_money = money;
                            best_action = { r1, c1, r2, c2 };
                        }
                        undo();
                    }
                }
            }
        }
        return best_action;
    }

    Action best_move() const {
        int lower_r = -1, lower_c = -1, lower_value = INT_MAX;
        int best_r = -1, best_c = -1, best_value = -1;
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                if (has_machine[r][c]) {
                    int value = (vege_board[r][c] ? vege_board[r][c]->value : 0);
                    if (value < lower_value) {
                        lower_value = value;
                        lower_r = r;
                        lower_c = c;
                    }
                }
            }
        }
        if (lower_r == INT_MAX) return { -1 };
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                if (!has_machine[r][c]) {
                    int value = (vege_board[r][c] ? vege_board[r][c]->value : 0);
                    if (best_value < value) {
                        best_value = value;
                        best_r = r;
                        best_c = c;
                    }
                }
            }
        }
        if (best_r == -1) return { -1 };
        if (best_value <= lower_value) return { -1 };
        return { lower_r, lower_c, best_r, best_c };
    }

    Action select_next_action() {
        if (turn >= 850) return best_move2();
        if (purchase_cost() <= money) {
            auto action = best_purchase();
            assert(action[0] != -1);
            return action;
        }
        return best_move();
    }

    void output(std::ostream& out) const {
        for (const auto& a : actions) {
            if (a.size() == 1) {
                out << a[0] << '\n';
                continue;
            }
            out << a[0] - 1;
            for (int i = 1; i < (int)a.size(); i++) out << ' ' << a[i] - 1;
            out << '\n';
        }
    }

#ifdef _MSC_VER
    cv::Mat_<cv::Vec3b> vis() const {
        static constexpr int GSIZE = 50;
        cv::Mat_<cv::Vec3b> img(GSIZE * (BOARD_SIZE + 2), GSIZE * (BOARD_SIZE + 2), cv::Vec3b(150, 150, 150));
        for (int row = 0; row < BOARD_SIZE + 2; row++) {
            for (int col = 0; col < BOARD_SIZE + 2; col++) {
                if (vege_board[row][col]) {
                    cv::Rect roi(col * GSIZE, row * GSIZE, GSIZE, GSIZE);
                    cv::rectangle(img, roi, cv::Scalar(255, 255, 255), cv::FILLED);
                    cv::putText(
                        img,
                        std::to_string(vege_board[row][col]->value),
                        cv::Point(roi.x + GSIZE / 5, roi.y + GSIZE / 2),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 0, 0)
                    );
                }
            }
        }
        return img;
    }
#endif
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
#ifdef _MSC_VER
    std::ifstream ifs("C:\\dev\\heuristic\\tasks\\RCL2021LongA\\tools\\tester\\input_0.txt");
    std::istream& in = ifs;
    std::ofstream ofs("C:\\dev\\heuristic\\tasks\\RCL2021LongA\\tools\\tester\\output_0.txt");
    std::ostream& out = ofs;
#else
    std::istream& in = std::cin;
    std::ostream& out = std::cout;
#endif

    { int buf; in >> buf >> buf >> buf; }

    std::vector<VegePtr> veges;
    for (int i = 0; i < NUM_VEGES; i++) {
        int R, C, S, E, V;
        in >> R >> C >> S >> E >> V;
        VegePtr vege = std::make_shared<Vege>(R + 1, C + 1, S, E, V);
        veges.push_back(vege);
    }

    State state(veges);

    while (state.turn < MAX_TURN) {
        auto action = state.select_next_action();
        state.simulate(action);
    }

    dump(state.money);

    state.output(out);

    dump(timer.elapsedMs());

    return 0;
}