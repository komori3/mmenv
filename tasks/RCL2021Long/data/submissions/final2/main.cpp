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
    Xorshift() {}
    Xorshift(unsigned seed) { set_seed(seed); }
    void set_seed(unsigned seed, int rep = 100) { x = (seed + 1) * 10007; for (int i = 0; i < rep; i++) next_int(); }
    unsigned next_int() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    unsigned next_int(unsigned mod) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % mod; }
    unsigned next_int(unsigned l, unsigned r) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % (r - l + 1) + l; }
    double next_double() { return double(next_int()) / UINT_MAX; }
} rnd;
template<typename T> void shuffle_vector(std::vector<T>& v, Xorshift& rnd) { int n = v.size(); for (int i = n - 1; i >= 1; i--) { int r = rnd.next_int(i); std::swap(v[i], v[r]); } }
std::vector<std::string> split(std::string str, const std::string& delim) { for (char& c : str) if (delim.find(c) != std::string::npos) c = ' '; std::istringstream iss(str); std::vector<std::string> parsed; std::string buf; while (iss >> buf) parsed.push_back(buf); return parsed; }

/* fast queue */
class FastQueue {
    int front, back;
    int v[1024];
public:
    FastQueue() : front(0), back(0) {}
    inline bool empty() { return front == back; }
    inline void push(int x) { v[front++] = x; }
    inline int pop() { return v[back++]; }
    inline void reset() { front = back = 0; }
    inline int size() { return front - back; }
};

using ll = long long;
using pii = std::pair<int, int>;
using pll = std::pair<ll, ll>;
using pdd = std::pair<double, double>;



constexpr int BOARD_SIZE = 16;
constexpr int NUM_VEGES = 5000;
constexpr int MAX_TURN = 1000;
constexpr int dr[] = { 0, -1, 0, 1 };
constexpr int dc[] = { 1, 0, -1, 0 };
constexpr int d4[] = { 1, -18, 18, -1 };

inline int enc(int r, int c) {
    return (r << 5) | c;
}
inline std::pair<int, int> dec(const int& rc) {
    return { rc >> 5, rc & 0b11111 };
}

struct Params;
using ParamsPtr = std::shared_ptr<Params>;
struct Params {
    Xorshift rnd;
    bool use_shuffle;
    double p[16];
};

struct Vege2;
using Vege2Ptr = Vege2*;
struct Vege2 {
    int rc;
    int begin, end;
    int value;
    Vege2(int rc = -1, int begin = -1, int end = -1, int value = -1)
        : rc(rc), begin(begin), end(end), value(value)
    {}

    static inline Vege2Ptr create_sentinel(int rc) {
        return new Vege2(rc, 2048, 4096, -1);
    }

    std::string str() const {
        return format("Vege [row=%d, col=%d, begin=%d, end=%d, value=%d]", rc / 18, rc % 18, begin, end, value);
    }
    friend std::ostream& operator<<(std::ostream& o, const Vege2& obj) {
        o << obj.str();
        return o;
    }
    friend std::ostream& operator<<(std::ostream& o, const Vege2Ptr& obj) {
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

struct Input2;
using Input2Ptr = std::shared_ptr<Input2>;
struct Input2 {
    std::vector<Vege2Ptr> veges;
    std::vector<std::vector<Vege2Ptr>> vege_begin;
    std::vector<std::vector<Vege2Ptr>> vege_end;
    Input2(std::istream& in) :
        veges(NUM_VEGES),
        vege_begin(MAX_TURN + 1),
        vege_end(MAX_TURN + 1)
    {
        { int buf; in >> buf >> buf >> buf; }
        for (int i = 0; i < NUM_VEGES; i++) {
            int R, C, S, E, V;
            in >> R >> C >> S >> E >> V;
            R++; C++;
            int RC = R * 18 + C;
            Vege2Ptr vege = new Vege2(RC, S, E, V); // TODO: 1-indexed or 0-indexed?
            veges[i] = vege;
            vege_begin[vege->begin].push_back(vege);
            vege_end[vege->end].push_back(vege);
        }
    }
};

struct State2;
using State2Ptr = std::shared_ptr<State2>;
struct State2 {
    using Action = pii;

    Input2Ptr input;

    std::vector<std::vector<Vege2Ptr>> vege_stack;
    
    int num_machines;
    std::vector<int> machine_pos;
    int has_machine[18 * 18];

    int distance_map[18 * 18];
    
    int turn;
    int money;

    // undo 用
    std::vector<Action> actions;
    std::vector<std::vector<Vege2Ptr>> removed;
    std::vector<int> money_hist;

    State2() {}
    State2(Input2Ptr input) :
        input(input),
        vege_stack(18 * 18),
        num_machines(0),
        machine_pos({ -1 }), // sentinel
        turn(0),
        money(1),
        removed(MAX_TURN),
        money_hist(MAX_TURN, -1)
    {
        memset(has_machine, 0, sizeof(int) * 18 * 18);
        for (int rc = 0; rc < 18 * 18; rc++) {
            vege_stack[rc].push_back(Vege2::create_sentinel(rc)); // TODO: 境界大丈夫？
        }
        for (int i = NUM_VEGES - 1; i >= 0; i--) {
            auto vege = input->veges[i];
            vege_stack[vege->rc].push_back(vege);
        }
    }

    inline int earned() const {
        int tmp = num_machines * (num_machines + 1) / 2;
        return money + tmp * tmp;
    }

    inline int calc_purchase_cost() const {
        return (num_machines + 1) * (num_machines + 1) * (num_machines + 1);
    }

    void purchase(int rc) {
        int cost = calc_purchase_cost();
        money -= cost;
        num_machines++;
        machine_pos.push_back(rc);
        has_machine[rc] = num_machines;
        actions.emplace_back(rc, -1);
    }

    void undo_purchase() {
        auto action = actions.back();
        actions.pop_back();
        int rc = action.first;
        has_machine[rc] = 0;
        machine_pos.pop_back();
        num_machines--;
        int cost = calc_purchase_cost();
        money += cost;
    }

    void move(int rc1, int rc2) {
        int mid = has_machine[rc1];
        machine_pos[mid] = rc2;
        has_machine[rc2] = mid;
        has_machine[rc1] = 0;
        actions.emplace_back(rc1, rc2);
    }

    void undo_move() {
        auto [rc1, rc2] = actions.back();
        actions.pop_back();
        int mid = has_machine[rc2];
        machine_pos[mid] = rc1;
        has_machine[rc1] = mid;
        has_machine[rc2] = 0;
    }

    void pass() {
        actions.emplace_back(-1, -1);
    }

    void undo_pass() {
        actions.pop_back();
    }

    void undo() {
        turn--;
        // harvest & disappear
        for (const auto& vege : removed[turn]) {
            vege_stack[vege->rc].push_back(vege);
        }
        removed[turn].clear();

        money = money_hist[turn];
        money_hist[turn] = -1;

        const auto& action = actions.back();
        if (action.second != -1) {
            undo_move();
        }
        else if (action.first != -1) {
            undo_purchase();
        }
        else {
            undo_pass();
        }
    }

    int count_connected_machines(int src) const {
        bool visited[18 * 18] = {};
        int cnt = 0;
        FastQueue fqu;
        fqu.push(src);
        visited[src] = true;
        cnt++;
        while (!fqu.empty()) {
            int rc = fqu.pop();
            for (int d = 0; d < 4; d++) {
                int nrc = rc + d4[d];
                if (visited[nrc] | !has_machine[nrc]) continue;
                fqu.push(nrc);
                visited[nrc] = true;
                cnt++;
            }
        }
        return cnt;
    }

    Vege2Ptr get_vege(int rc) const {
        const auto& vege = vege_stack[rc].back();
        return (vege->begin <= turn) ? vege : nullptr;
    }

    Action purchase_1() const {
        std::vector<std::tuple<int, int, int>> tup; // (-value, end, rc)
        tup.emplace_back(1000, -1, -1); // pass
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                int rc = r * 18 + c;
                auto vege = get_vege(rc);
                if (!vege) continue;
                tup.emplace_back(-vege->value, vege->end, rc);
            }
        }
        std::sort(tup.begin(), tup.end());
        return { std::get<2>(tup.front()), -1 };
    }

    Action move_1() const {
        int rc1 = machine_pos[1];
        std::vector<std::tuple<int, int, int>> tup; // (-value, end, rc)
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                int rc = r * 18 + c;
                auto vege = get_vege(rc);
                if (!vege) continue;
                tup.emplace_back(-vege->value, vege->end, rc);
            }
        }
        tup.emplace_back(1000, -1, -1); // pass
        std::sort(tup.begin(), tup.end());
        auto [_, ___, rc2] = tup.front();
        if (rc1 == rc2) {
            return { -1, -1 };
        }
        return { rc1, rc2 };
    }

    std::vector<int> enum_outer_contour_points() const {
        bool inflated[18 * 18] = {};
        for (int i = 0; i < BOARD_SIZE + 2; i++) {
            inflated[i] = inflated[17 * 18 + i] = inflated[i * 18] = inflated[i * 18 + 17] = true;
        }
        std::vector<int> contour;
        for (int mid = 1; mid <= num_machines; mid++) {
            int rc = machine_pos[mid];
            for (int d = 0; d < 4; d++) {
                int nrc = rc + d4[d];
                if (!inflated[nrc] && !has_machine[nrc]) {
                    inflated[nrc] = true;
                    contour.emplace_back(nrc);
                }
            }
        }
        return contour;
    }

    std::vector<Action> enum_purchase_actions() {
        std::vector<Action> actions;
        for (int rc : enum_outer_contour_points()) {
            actions.emplace_back(rc, -1);
        }
        return actions;
    }

    std::vector<Action> enum_connective_move_actions() {
        std::vector<Action> actions;
        auto contour = enum_outer_contour_points(); // 外輪郭
        for (int rc1 : machine_pos) {
            if (rc1 == -1) continue;
            int mid = has_machine[rc1];
            has_machine[rc1] = 0;
            for (int rc2 : contour) {
                has_machine[rc2] = mid;
                if (count_connected_machines(rc2) == num_machines) {
                    actions.emplace_back(rc1, rc2);
                }
                has_machine[rc2] = 0;
            }
            has_machine[rc1] = mid;
        }
        return actions;
    }

    void update_distance_map() {
        static constexpr int inf = 100;
        for (int n = 0; n < 18 * 18; n++) distance_map[n] = inf;
        for (int i = 0; i < BOARD_SIZE + 2; i++) {
            distance_map[i * 18] = distance_map[i * 18 + 17] = distance_map[i] = distance_map[17 * 18 + i] = 0;
        }
        FastQueue fqu;
        for (int mid = 1; mid <= num_machines; mid++) {
            int rc = machine_pos[mid];
            distance_map[rc] = 0;
            fqu.push(rc);
        }
        while (!fqu.empty()) {
            int rc = fqu.pop();
            for (int d = 0; d < 4; d++) {
                int nrc = rc + d4[d];
                if (distance_map[nrc] == inf) {
                    distance_map[nrc] = distance_map[rc] + 1;
                    fqu.push(nrc);
                }
            }
        }
    }

    double evaluate(const Action& action, ParamsPtr params) {
        double score = 0;

        if (action.second != -1) {
            auto [rc1, rc2] = action;
            int mid = has_machine[rc1];
            machine_pos[mid] = rc2;
            has_machine[rc2] = mid;
            has_machine[rc1] = 0;
        }
        else if (action.first != -1) {
            auto [rc, _] = action;
            num_machines++;
            machine_pos.push_back(rc);
            has_machine[rc] = num_machines;
        }

        int margin = (int)round(num_machines * params->p[0]);

        update_distance_map();
        for (int r = 1; r <= BOARD_SIZE; r++) {
            for (int c = 1; c <= BOARD_SIZE; c++) {
                int rc = r * 18 + c;
                if (has_machine[rc]) {
                    auto vege = vege_stack[rc].back();
                    if ((vege->begin + vege->end) / 2 <= turn + margin) {
                        double priority = params->p[1];
                        score += vege->value * priority;
                    }
                }
                else {
                    auto vege = vege_stack[rc].back();
                    if ((vege->begin + vege->end) / 2 <= turn + margin) {
                        int d = distance_map[rc];
                        if (vege->end - turn >= d) {
                            double priority = params->p[2] / d;
                            score += vege->value * priority;
                        }
                        else {
                            double priority = params->p[3] / d;
                            score -= vege->value * priority;
                        }
                    }
                }
            }
        }

        if (action.second != -1) {
            auto [rc1, rc2] = action;
            int mid = has_machine[rc2];
            machine_pos[mid] = rc1;
            has_machine[rc1] = mid;
            has_machine[rc2] = 0;
        }
        else if (action.first != -1) {
            auto [rc, _] = action;
            num_machines--;
            machine_pos.pop_back();
            has_machine[rc] = 0;
        }

        return score;
    }

    Action select_next_action(ParamsPtr params) {
        if (num_machines == 0) {
            return purchase_1();
        }
        if (num_machines == 1 && money < calc_purchase_cost()) {
            return move_1();
        }
        if (num_machines < 50 && calc_purchase_cost() <= money) {
            auto actions = enum_purchase_actions();
            if (params->use_shuffle) shuffle_vector(actions, params->rnd);
            double best = -1e9;
            Action best_action;
            for (const auto& action : actions) {
                double score = evaluate(action, params);
                if (best < score) {
                    best = score;
                    best_action = action;
                }
            }
            return best_action;
        }
        auto actions = enum_connective_move_actions();
        actions.emplace_back(-1, -1);
        if (params->use_shuffle) shuffle_vector(actions, params->rnd);
        double best = -1e9;
        Action best_action;
        for (const auto& action : actions) {
            double score = evaluate(action, params);
            if (best < score) {
                best = score;
                best_action = action;
            }
        }
        return best_action;
    }

    void simulate(const Action& action) {
        // action
        if (action.second != -1) {
            move(action.first, action.second);
        }
        else if (action.first != -1) {
            purchase(action.first);
        }
        else {
            pass();
        }

        money_hist[turn] = money;

        // calc score
        for (int mid = 1; mid <= num_machines; mid++) {
            int rc = machine_pos[mid];
            if (!has_machine[rc]) continue;
            auto vege = get_vege(rc);
            if (vege) {
                money += vege->value * count_connected_machines(rc);
                removed[turn].push_back(vege);
                vege_stack[rc].pop_back();
            }
        }
        // disappear
        for (const auto& vege : input->vege_end[turn]) {
            if (get_vege(vege->rc) == vege) {
                removed[turn].push_back(vege);
                vege_stack[vege->rc].pop_back();
            }
        }
        turn++;
    }

    void solve() {
        ParamsPtr params = std::make_shared<Params>();
        params->rnd = Xorshift(0);
        params->use_shuffle = false;
        params->p[0] = 0.5;
        params->p[1] = 1.0;
        params->p[2] = 0.5;
        params->p[3] = params->p[2] * 1.5;

        while (turn < 800) {
            auto action = select_next_action(params);
            simulate(action);
        }

        std::vector<pii> ranges({ {800, 850}, {850, 900}, {900, 950}, {950, 1000} });

        for (auto [begin, end] : ranges) {
            int best_earned = 0;
            std::vector<pii> best_actions;
            for (double p2 : {0.9, 1.0, 1.1}) {
                params->p[0] = 0.5;
                params->p[1] = p2;
                params->p[2] = 0.5;
                params->p[3] = params->p[2] * 1.5;
                while (turn < end) {
                    auto action = select_next_action(params);
                    simulate(action);
                }
                if (best_earned < earned()) {
                    best_earned = earned();
                    auto& as = actions;
                    best_actions = std::vector<pii>(as.begin() + begin, as.begin() + end);
                }
                while (turn > begin) undo();
            }
            for (const auto& action : best_actions) simulate(action);
        }
    }

    void output(std::ostream& out) const {
        for (const auto& a : actions) {
            if (a.second != -1) {
                out << a.first / 18 - 1 << ' '
                    << a.first % 18 - 1 << ' '
                    << a.second / 18 - 1 << ' '
                    << a.second % 18 - 1 << '\n';
            }
            else if (a.first != -1) {
                out << a.first / 18 - 1 << ' '
                    << a.first % 18 - 1 << '\n';
            }
            else {
                out << -1 << '\n';
            }
        }
    }
};

#ifdef _MSC_VER
int _main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    std::vector<int> scores(51, 0);
    concurrency::parallel_for(1, 51, [&scores](int seed) {
        std::ifstream ifs(format("C:\\dev\\heuristic\\tasks\\RCL2021Long\\tools\\tester\\in\\%d.in", seed));
        std::istream& in = ifs;
        Input2Ptr input = std::make_shared<Input2>(in);
        State2 state(input);
        state.solve();
        scores[seed] = state.money;
        dump(state.money);
    });

    dump(scores);
    dump(std::accumulate(scores.begin(), scores.end(), 0));
    return 0;
}
#endif

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
#ifdef _MSC_VER
    std::ifstream ifs("C:\\dev\\heuristic\\tasks\\RCL2021Long\\tools\\tester\\in\\534.in");
    std::istream& in = ifs;
    std::ofstream ofs("C:\\dev\\heuristic\\tasks\\RCL2021Long\\tools\\tester\\out\\534.out");
    std::ostream& out = ofs;
#else
    std::istream& in = std::cin;
    std::ostream& out = std::cout;
#endif

    Input2Ptr input = std::make_shared<Input2>(in);

    State2 state(input);

    state.solve();

    dump(state.money, state.earned());
    dump(timer.elapsedMs());

    state.output(out);

    return 0;
}