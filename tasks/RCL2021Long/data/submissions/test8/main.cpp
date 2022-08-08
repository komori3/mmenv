#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <unordered_set>
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
constexpr int dr[] = { 0, -1, 0, 1 };
constexpr int dc[] = { 1, 0, -1, 0 };

inline int enc(int r, int c) {
    return (r << 5) | c;
}
inline std::pair<int, int> dec(const int& rc) {
    return { rc >> 5, rc & 0b11111 };
}

struct Vege;
using VegePtr = Vege*;
struct Vege {
    int row, col;
    int begin, end;
    int value;
    Vege(int row = -1, int col = -1, int begin = -1, int end = -1, int value = -1)
        : row(row), col(col), begin(begin), end(end), value(value)
    {}

    static inline VegePtr create_sentinel(int r, int c) {
        return new Vege(r, c, 2048, 4096, -1);
    }

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

struct Cell {
    int row, col;
    int value;
    int duration;
    int next_value;
    int wait_time;
    bool has_machine;
    bool purchased;
    bool harvested;
    bool moved;
    int rsrc, csrc;
};

struct State;
using StatePtr = std::shared_ptr<State>;
struct State {
    using Action = std::vector<int>;

    std::vector<VegePtr> veges;
    std::vector<std::vector<VegePtr>> vege_begin;
    std::vector<std::vector<VegePtr>> vege_end;

    std::vector<std::vector<std::vector<VegePtr>>> vege_stack;

    int num_machines;
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
        vege_stack(BOARD_SIZE + 2, std::vector<std::vector<VegePtr>>(BOARD_SIZE + 2)),
        num_machines(0),
        has_machine(BOARD_SIZE + 2, std::vector<bool>(BOARD_SIZE + 2, false)),
        turn(0),
        money(1),
        removed(MAX_TURN),
        money_hist(MAX_TURN, -1)
    {
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                vege_stack[r][c].push_back(Vege::create_sentinel(r, c));
            }
        }
        for (int i = veges.size() - 1; i >= 0; i--) {
            auto vege = veges[i];
            vege_begin[vege->begin].push_back(vege);
            vege_end[vege->end].push_back(vege);
            vege_stack[vege->row][vege->col].push_back(vege);
        }
    }

    inline int calc_purchase_cost() const {
        return (num_machines + 1) * (num_machines + 1) * (num_machines + 1);
    }

    void purchase(int r, int c) {
        int cost = calc_purchase_cost();
        assert(cost <= money);
        money -= cost;
        num_machines++;
        has_machine[r][c] = true;
        actions.push_back({ r, c });
    }

    void undo_purchase() {
        auto action = actions.back();
        actions.pop_back();
        int r = action[0], c = action[1];
        has_machine[r][c] = false;
        num_machines--;
        int cost = calc_purchase_cost();
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

    int count_connected_machines(int sr, int sc) const {
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
                if (!has_machine[r][c]) continue;
                auto vege = get_vege(r, c);
                if (vege) {
                    money += vege->value * count_connected_machines(r, c);
                    removed[turn].push_back(vege);
                    vege_stack[r][c].pop_back();
                }
            }
        }
        // disappear
        for (const auto& vege : vege_end[turn]) {
            if (get_vege(vege->row, vege->col) == vege) {
                removed[turn].push_back(vege);
                vege_stack[vege->row][vege->col].pop_back();
            }
        }
        turn++;
    }

    void undo() {
        turn--;
        // harvest & disappear
        for (const auto& vege : removed[turn]) {
            vege_stack[vege->row][vege->col].push_back(vege);
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

    int calc_vege_value(int r, int c) const {
        auto vege = vege_stack[r][c].back();
        return (vege->begin <= turn) ? vege->value : 0;
    }

    VegePtr get_vege(int r, int c) const {
        const auto& vege = vege_stack[r][c].back();
        return (vege->begin <= turn) ? vege : nullptr;
    }

    Action best_purchase() {
        int best_r = -1, best_c = -1, best_value = -1;
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                if (has_machine[r][c]) continue;
                has_machine[r][c] = true;
                int value = calc_vege_value(r, c) * count_connected_machines(r, c);
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
                    int value = calc_vege_value(r, c);
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
                    int value = calc_vege_value(r, c);
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

    Action purchase_1() const {
        std::vector<std::tuple<int, int, int>> tup;
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                auto vege = get_vege(r, c);
                if (!vege) continue;
                tup.emplace_back(vege->end, r, c);
            }
        }
        std::sort(tup.begin(), tup.end());
        return { std::get<1>(tup.front()), std::get<2>(tup.front()) };
    }

    Action move_1() const {
        int r1 = -1, c1 = -1;
        std::vector<std::tuple<int, int, int>> tup;
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                if (has_machine[r][c]) {
                    r1 = r; 
                    c1 = c;
                }
                auto vege = get_vege(r, c);
                if (!vege) continue;
                tup.emplace_back(vege->end, r, c);
            }
        }
        std::sort(tup.begin(), tup.end());
        auto [_, r2, c2] = tup.front();
        if (r1 == r2 && c1 == c2) {
            return { -1 };
        }
        return { r1, c1, r2, c2 };
    }

    std::vector<pii> enum_outer_contour_points() const {
        std::vector<std::vector<bool>> inflated(BOARD_SIZE + 2, std::vector<bool>(BOARD_SIZE + 2, false));
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                if (!has_machine[r][c]) continue;
                for (int d = 0; d < 4; d++) {
                    inflated[r + dr[d]][c + dc[d]] = true;
                }
            }
        }
        std::vector<pii> contour;
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                if (!has_machine[r][c] && inflated[r][c]) {
                    contour.push_back({ r, c });
                }
            }
        }
        return contour;
    }

    std::vector<Action> enum_purchase_actions() {
        std::vector<Action> actions;
        for (const auto& [r, c] : enum_outer_contour_points()) {
            actions.push_back({ r, c });
        }
        return actions;
    }

    std::vector<Action> enum_connective_move_actions() {
        std::vector<Action> actions;
        auto machine = enum_machine_pos();
        auto contour = enum_outer_contour_points(); // 外輪郭
        for (auto [r1, c1] : machine) {
            has_machine[r1][c1] = false;
            for (auto [r2, c2] : contour) {
                has_machine[r2][c2] = true;
                if (count_connected_machines(r2, c2) == num_machines) {
                    actions.push_back({ r1, c1, r2, c2 });
                }
                has_machine[r2][c2] = false;
            }
            has_machine[r1][c1] = true;
        }
        return actions;
    }

    double evaluate(const Action& action) {
        double score = DBL_MIN;

        if (action.size() == 2) {
            int r = action[0], c = action[1];
            has_machine[r][c] = true;
        }
        else if (action.size() == 4) {
            int r1 = action[0], c1 = action[1], r2 = action[2], c2 = action[3];
            has_machine[r1][c1] = false;
            has_machine[r2][c2] = true;
        }

        int margin = num_machines / 2;

        auto machine_pos = enum_machine_pos();
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                if (has_machine[r][c]) {
                    auto vege = vege_stack[r][c].back();
                    if (vege->begin <= turn + margin) {
                        double priority = 1.0 / std::max(vege->begin - turn, 1);
                        //double priority = 1.0 / (vege->end - turn + 1);
                        score += vege->value * priority;
                    }
                }
                else {
                    auto vege = vege_stack[r][c].back();
                    if (vege->begin <= turn + margin) {
                        int mindist = INT_MAX;
                        for (auto [mr, mc] : machine_pos) {
                            mindist = std::min(mindist, abs(mr - r) + abs(mc - c));
                        }
                        if (vege->end - turn >= mindist) {
                            double priority = 0.5 / mindist;
                            score += vege->value * priority;
                        }
                    }
                }
            }
        }

        if (action.size() == 2) {
            int r = action[0], c = action[1];
            has_machine[r][c] = false;
        }
        else if (action.size() == 4) {
            int r1 = action[0], c1 = action[1], r2 = action[2], c2 = action[3];
            has_machine[r1][c1] = true;
            has_machine[r2][c2] = false;
        }

        return score;
    }

    Action select_next_action() {
        if (num_machines == 0) {
            return purchase_1();
        }
        if (num_machines == 1 && money < calc_purchase_cost()) {
            return move_1();
        }
        //if (turn < 850 && calc_purchase_cost() <= money) {
        if (num_machines < 47 && calc_purchase_cost() <= money) {
            auto actions = enum_purchase_actions();
            double best = -1;
            Action best_action;
            for (const auto& action : actions) {
                double score = evaluate(action);
                if (best < score) {
                    best = score;
                    best_action = action;
                }
            }
            return best_action;
        }
        auto actions = enum_connective_move_actions();
        actions.push_back({ -1 });
        double best = -1;
        Action best_action;
        for (const auto& action : actions) {
            double score = evaluate(action);
            if (best < score) {
                best = score;
                best_action = action;
            }
        }
        return best_action;
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

    std::vector<pii> enum_machine_pos() const {
        std::vector<pii> pos;
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                if (has_machine[r][c]) {
                    pos.emplace_back(r, c);
                }
            }
        }
        return pos;
    }

    std::vector<std::vector<int>> get_value_sum_map(int begin = 0, int end = MAX_TURN) const {
        std::vector<std::vector<int>> value_map(BOARD_SIZE + 2, std::vector<int>(BOARD_SIZE + 2, 0));
        for (const auto& vege : veges) {
            if (begin < vege->begin || vege->end < end) continue;
            value_map[vege->row][vege->col] += vege->value;
        }
        return value_map;
    }

#ifdef _MSC_VER
    cv::Mat_<cv::Vec3b> get_board_image() const {
        static const cv::Scalar LIGHT_GREEN(223, 255, 223);
        static const cv::Scalar GREEN(95, 255, 95);
        static const cv::Scalar BROWN(11, 134, 184);
        static const cv::Scalar FADE_BROWN(181, 231, 215);
        static const cv::Scalar RED(45, 92, 249);
        static const cv::Scalar BLACK(0, 0, 0);
        static const cv::Scalar GRAY(150, 150, 150);
        static constexpr int MARGIN = 10;
        static constexpr int GSIZE = 80;
        static constexpr int HEIGHT = MARGIN * 2 + GSIZE * BOARD_SIZE;
        static constexpr int WIDTH = HEIGHT;
        auto get_point = [&](int r, int c) {
            return cv::Point(MARGIN + (c - 1) * GSIZE, MARGIN + (r - 1) * GSIZE);
        };
        auto get_rect = [&](int r, int c) {
            return cv::Rect(MARGIN + (c - 1) * GSIZE, MARGIN + (r - 1) * GSIZE, GSIZE, GSIZE);
        };

        cv::Mat_<cv::Vec3b> img(HEIGHT, WIDTH, cv::Vec3b(LIGHT_GREEN[0], LIGHT_GREEN[1], LIGHT_GREEN[2]));

        std::vector<std::vector<Cell>> cells(BOARD_SIZE + 2, std::vector<Cell>(BOARD_SIZE + 2));
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                auto& cell = cells[r][c];
                cell.row = r;
                cell.col = c;
                cell.has_machine = has_machine[r][c];
                auto vege = get_vege(r, c);
                if (vege) {
                    cell.value = vege->value;
                    cell.duration = vege->end - turn + 1;
                    cell.next_value = -1;
                    cell.wait_time = -1;
                }
                else {
                    auto next_vege = vege_stack[r][c].back();
                    cell.value = -1;
                    cell.duration = -1;
                    cell.next_value = next_vege->value;
                    cell.wait_time = next_vege->begin - turn;
                }
            }
        }
        
        if (turn) {
            // 消えたばかりのものは追加
            for (const auto& vege : removed[turn - 1]) {
                auto& cell = cells[vege->row][vege->col];
                cell.value = vege->value;
                cell.duration = vege->end - turn + 1;
                cell.next_value = -1;
                cell.wait_time = -1;
                // harvest: has_machine かつ removed に存在
                if (has_machine[vege->row][vege->col]) {
                    cell.harvested = true;
                }
            }
        }
        // 現れたばかりのものは除外
        for (const auto& vege : vege_begin[turn]) {
            auto& cell = cells[vege->row][vege->col];
            cell.harvested = false;
            cell.value = -1;
            cell.duration = -1;
            cell.next_value = vege->value;
            cell.wait_time = vege->begin - turn;
        }

        if (turn) {
            // purchase
            if (actions.back().size() == 2) {
                int r = actions.back()[0], c = actions.back()[1];
                cells[r][c].purchased = true;
            }
            // move
            if (actions.back().size() == 4) {
                const auto& action = actions.back();
                int r1 = action[0], c1 = action[1], r2 = action[2], c2 = action[3];
                cells[r2][c2].moved = true;
                cells[r2][c2].rsrc = r1; cells[r2][c2].csrc = c1;
            }
        }

        for (int row = 1; row <= BOARD_SIZE; row++) {
            for (int col = 1; col <= BOARD_SIZE; col++) {
                auto& cell = cells[row][col];
                auto roi = get_rect(row, col);
                if (cell.harvested) {
                    cv::rectangle(img, roi, GREEN, cv::FILLED);
                }
                if (cell.has_machine) {
                    cv::Point p1(roi.x + 3, roi.y + 3), p2(roi.x + roi.width - 3, roi.y + roi.height - 3);
                    int thickness = (cell.moved || cell.purchased) ? 5 : 2;
                    cv::Scalar color = cell.purchased ? RED : BROWN;
                    cv::rectangle(img, p1, p2, color, thickness);
                    if (cell.moved) {
                        auto roi2 = get_rect(cell.rsrc, cell.csrc);
                        cv::Point p3(roi2.x + 3, roi2.y + 3), p4(roi2.x + roi2.width - 3, roi2.y + roi2.height - 3);
                        cv::rectangle(img, p3, p4, FADE_BROWN, 2);
                        cv::Point cdst(roi.x + roi.width / 2, roi.y + roi.height / 2);
                        cv::Point csrc(roi2.x + roi2.width / 2, roi2.y + roi2.height / 2);
                        cv::arrowedLine(img, csrc, cdst, BROWN);
                    }
                }
                if (cell.value != -1) {
                    cv::Rect roi = get_rect(row, col);
                    cv::putText(
                        img,
                        format("%d", cell.value),
                        cv::Point(roi.x + GSIZE / 5, roi.y + GSIZE * 3 / 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        cv::Scalar(0, 0, 0),
                        2
                    );
                    cv::putText(
                        img,
                        format("%d", cell.duration),
                        cv::Point(roi.x + GSIZE * 2 / 3, roi.y + GSIZE / 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.4,
                        cv::Scalar(0, 0, 0)
                    );
                }
                else if (cell.wait_time <= 10) {
                    cv::Rect roi = get_rect(row, col);
                    cv::Scalar color;
                    {
                        double g = cell.wait_time / 10.0, b = (10 - cell.wait_time) / 10.0;
                        color = LIGHT_GREEN * g + GRAY * b;
                    }
                    cv::putText(
                        img,
                        format("%d", cell.next_value),
                        cv::Point(roi.x + GSIZE / 5, roi.y + GSIZE * 3 / 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    );
                    cv::putText(
                        img,
                        format("%d", cell.wait_time),
                        cv::Point(roi.x + GSIZE / 6, roi.y + GSIZE / 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color
                    );
                }
            }
        }

        for (int n = 1; n <= BOARD_SIZE + 1; n++) {
            cv::line(img, get_point(n, 1), get_point(n, BOARD_SIZE + 1), cv::Scalar(150, 150, 150));
            cv::line(img, get_point(1, n), get_point(BOARD_SIZE + 1, n), cv::Scalar(150, 150, 150));
        }

        return img;
    }
    cv::Mat_<cv::Vec3b> get_image() const {
        cv::Mat_<cv::Vec3b> img_board = get_board_image();
        int HEIGHT = img_board.rows;
        int BOARD_WIDTH = img_board.cols;
        int INFO_WIDTH = 300;
        int WIDTH = BOARD_WIDTH + INFO_WIDTH;

        cv::Mat_<cv::Vec3b> img_info(HEIGHT, INFO_WIDTH, cv::Vec3b(255, 255, 255));
        {
            cv::putText(img_info, format("turn: %d", turn), cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            cv::putText(img_info, format("money: %d", money), cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            std::ostringstream oss;
            oss << (actions.size() ? actions.back() : std::vector<int>());
            cv::putText(img_info, format("action: %s", oss.str().c_str()), cv::Point(20, 150), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            cv::putText(img_info, format("machines: %d", num_machines), cv::Point(20, 200), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            cv::putText(img_info, format("price: %d", calc_purchase_cost()), cv::Point(20, 250), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }

        cv::Mat_<cv::Vec3b> img(HEIGHT, WIDTH);
        {
            cv::Rect roi(0, 0, BOARD_WIDTH, HEIGHT);
            img_board.copyTo(img(roi));
        }
        {
            cv::Rect roi(BOARD_WIDTH, 0, INFO_WIDTH, HEIGHT);
            img_info.copyTo(img(roi));
        }
        return img;
    }
    pii calc_cell_pos(int x, int y) const {
        static constexpr int MARGIN = 10;
        static constexpr int GSIZE = 80;
        static constexpr int HEIGHT = MARGIN * 2 + GSIZE * BOARD_SIZE;
        static constexpr int WIDTH = HEIGHT;
        if (x < MARGIN || MARGIN + GSIZE * BOARD_SIZE <= x) return { -1, -1 };
        if (y < MARGIN || MARGIN + GSIZE * BOARD_SIZE <= y) return { -1, -1 };
        return { (y - MARGIN) / GSIZE + 1, (x - MARGIN) / GSIZE + 1 };
    }
#endif
};

#ifdef _MSC_VER

struct MouseParams;
using MouseParamsPtr = std::shared_ptr<MouseParams>;
struct MouseParams {
    int pe, px, py, pf;
    int e, x, y, f;
    MouseParams(int pe = -1, int px = -1, int py = -1, int pf = -1, int e = -1, int x = -1, int y = -1, int f = -1)
        : pe(pe), px(px), py(py), pf(pf), e(e), x(x), y(y), f(f) {}
    inline void load(int e_, int x_, int y_, int f_) {
        pe = e; px = x; py = y; pf = f;
        e = e_; x = x_; y = y_; f = f_;
    }
    inline bool clicked_left() const { return e == 1 && pe == 0; }
    inline bool clicked_right() const { return e == 2 && pe == 0; }
    std::string str() const {
        return format("MouseParams [(pe,px,py,pf)=(%d,%d,%d,%d), (e,x,y,f)=(%d,%d,%d,%d)]", pe, px, py, pf, e, x, y, f);
    }
    friend std::ostream& operator<<(std::ostream& o, const MouseParams& obj) {
        o << obj.str();
        return o;
    }
    friend std::ostream& operator<<(std::ostream& o, const MouseParamsPtr& obj) {
        o << obj->str();
        return o;
    }
};

struct ManualSolver;
using ManualSolverPtr = std::shared_ptr<ManualSolver>;
struct ManualSolver {
    std::string window_name;
    MouseParamsPtr mp;
    StatePtr state;
    pii selected;

    ManualSolver(StatePtr state) : window_name("manual"), state(state), selected(-1, -1) {
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        mp = std::make_shared<MouseParams>();
        cv::setMouseCallback(window_name, callback, this);
    }

    void play() {
        while (state->turn < MAX_TURN) {
            cv::imshow(window_name, state->get_image());
            int c = cv::waitKey(15);
            if (c == 'p') {
                state->simulate({ -1 });
            }
        }
    }

    static void callback(int e, int x, int y, int f, void* param) {
        ManualSolver* sol = static_cast<ManualSolver*>(param);
        auto state = sol->state;
        auto mp = sol->mp;
        mp->load(e, x, y, f);
        if (mp->clicked_left()) {
            auto [r, c] = state->calc_cell_pos(x, y);
            if (r == -1) return;
            if (sol->selected.first == -1) {
                if (!state->has_machine[r][c]) {
                    if (state->calc_purchase_cost() <= state->money) {
                        std::vector<int> action({ r, c });
                        state->simulate(action);
                        return;
                    }
                }
                else {
                    if (state->has_machine[r][c]) {
                        sol->selected = pii(r, c);
                        return;
                    }
                }
            }
            else {
                if (!state->has_machine[r][c]) {
                    std::vector<int> action({ sol->selected.first, sol->selected.second, r, c });
                    state->simulate(action);
                    sol->selected = pii(-1, -1);
                    return;
                }
            }
        }
        if (mp->clicked_right()) {
            if (sol->selected.first != -1) {
                sol->selected = pii(-1, -1);
                return;
            }
            else {
                if (state->turn) {
                    state->undo();
                    return;
                }
            }
        }
    }
};
#endif

#ifdef _MSC_VER
void solve_manual(const std::vector<VegePtr>& veges, std::ostream& out) {
    StatePtr state = std::make_shared<State>(veges);
    ManualSolverPtr sol = std::make_shared<ManualSolver>(state);
    sol->play();
    state->output(out);
}
#endif

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
#ifdef _MSC_VER
    std::ifstream ifs("C:\\dev\\heuristic\\tasks\\RCL2021Long\\tools\\tester\\input_0.txt");
    std::istream& in = ifs;
    std::ofstream ofs("C:\\dev\\heuristic\\tasks\\RCL2021Long\\tools\\tester\\output_0.txt");
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
        VegePtr vege = new Vege(R + 1, C + 1, S, E, V);
        veges.push_back(vege);
    }

    //solve_manual(veges, out);

    StatePtr state = std::make_shared<State>(veges);
    while (state->turn < MAX_TURN) {
        auto action = state->select_next_action();
        //std::cerr << action << std::endl;
        state->simulate(action);
        //cv::imshow("img", state->get_image());
        //cv::waitKey(0);
    }
    //cv::imshow("img", state->get_image());
    //cv::waitKey(0);
    dump(state->money);
    state->output(out);

    dump(timer.elapsedMs());

    return 0;
}