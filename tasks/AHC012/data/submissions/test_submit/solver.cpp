#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#ifdef _MSC_VER
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <conio.h>
#include <ppl.h>
#include <filesystem>
#include <intrin.h>
//#include <boost/multiprecision/cpp_int.hpp>
int __builtin_clz(unsigned int n)
{
    unsigned long index;
    _BitScanReverse(&index, n);
    return 31 - index;
}
int __builtin_ctz(unsigned int n)
{
    unsigned long index;
    _BitScanForward(&index, n);
    return index;
}
namespace std {
    inline int __lg(int __n) { return sizeof(int) * 8 - 1 - __builtin_clz(__n); }
}
//using __uint128_t = boost::multiprecision::uint128_t;
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

std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) {
    std::string s(v.size(), ' ');
    for (int i = 0; i < v.size(); i++) s[i] = v[i] + '0';
    os << s;
    return os;
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
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }

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
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
};

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

using ll = long long;
using ld = double;
//using ld = boost::multiprecision::cpp_bin_float_quad;
using pii = std::pair<int, int>;
using pll = std::pair<ll, ll>;

using std::cin, std::cout, std::cerr, std::endl, std::string, std::vector;



constexpr int offset = 10000;
// 座標は +10000 して扱う
// x, y は [1, 19999] に収まる

struct Input {
    int N, K;
    int a[10];
    vector<pii> xys;
    vector<vector<int>> y2xs;
    vector<vector<int>> x2ys;
    Input(std::istream& in) {
        in >> N >> K >> a;
        xys.resize(N);
        y2xs.resize(20001);
        x2ys.resize(20001);
        in >> xys;
        for (auto& [x, y] : xys) {
            x += offset;
            y += offset;
            y2xs[y].push_back(x);
            x2ys[x].push_back(y);
        }
    }
    string stringify() const {
        string res;
        res += format("N=%d, K=%d\n", N, K);
        res += "a=[";
        for (int i = 0; i < 10; i++) res += std::to_string(a[i]) + ',';
        res += "]\n";
        res += "xys=[";
        for (const auto& [x, y] : xys) {
            res += format("(%d,%d),", x, y);
        }
        res += "]\n";
        return res;
    }
};

using Output = std::vector<std::tuple<int, int, int, int>>;

struct Input2 {
    int N, K;
    int a[10];
    vector<pii> xys;
    Input2(std::istream& in) {
        in >> N >> K >> a;
        xys.resize(N);
        in >> xys;
    }
};

struct Trans {
    int id, dir, dist, diff;
    Trans(int id, int dir, int dist, int diff) : id(id), dir(dir), dist(dist), diff(diff) {}
};

template<typename T>
struct Compress {
    std::vector<T> to_value;
    std::unordered_map<T, int> to_index;
    Compress(const std::vector<T>& v) {
        std::set<T> st(v.begin(), v.end());
        to_value = std::vector<T>(st.begin(), st.end());
        for (int i = 0; i < (int)to_value.size(); i++) to_index[to_value[i]] = i;
    }
};

struct State2 {

    const Input2& input;

    int H, W; // 縦線・横線
    int target[1001]; // 最大 1000 個
    int now[1001];

    int cost;

    vector<int> xs, ys;
    std::unordered_map<int, int> x2c, y2r;

    vector<int> rpos, cpos;
    vector<int> r2id, c2id;
    vector<vector<int>> r2cs, c2rs;

    vector<vector<int>> area_points;

    State2(const Input2& input, int H, int W) : input(input), H(H), W(W) {

        memset(target, 0, sizeof(int) * 1001);
        for (int d = 0; d < 10; d++) target[d + 1] = input.a[d];
        memset(now, 0, sizeof(int) * 1001);

        xs.push_back(-10010); xs.push_back(10010);
        ys.push_back(-10010); ys.push_back(10010);
        for (const auto& [x, y] : input.xys) {
            xs.push_back(x);
            ys.push_back(y);
        }
        Compress<int> xcmp(xs), ycmp(ys);
        xs = xcmp.to_value;
        x2c = xcmp.to_index;
        ys = ycmp.to_value;
        y2r = ycmp.to_index;

        cost = 0;
        
        int rdim = ys.size(), cdim = xs.size();
        int row_interval = rdim / (H + 1), col_interval = cdim / (W + 1);
        rpos.resize(H + 2);
        rpos[0] = 0; rpos[H + 1] = rdim - 1;
        for (int i = 1; i <= H; i++) {
            int r = row_interval * i;
            rpos[i] = r;
        }
        cpos.resize(W + 2);
        cpos[0] = 0; cpos[W + 1] = cdim - 1;
        for (int j = 1; j <= W; j++) {
            int c = col_interval * j;
            cpos[j] = c;
        }

        r2id.resize(rdim);
        for (int rid = 0; rid + 1 < rpos.size(); rid++) {
            for (int r = rpos[rid]; r < rpos[rid + 1]; r++) {
                r2id[r] = rid;
            }
        }

        c2id.resize(cdim);
        for (int cid = 0; cid + 1 < cpos.size(); cid++) {
            for (int c = cpos[cid]; c < cpos[cid + 1]; c++) {
                c2id[c] = cid;
            }
        }

        r2cs.resize(rdim);
        c2rs.resize(cdim);
        area_points.resize(H + 1, vector<int>(W + 1, 0));
        for (const auto& [x, y] : input.xys) {
            int r = y2r[y], c = x2c[x];
            r2cs[r].push_back(c);
            c2rs[c].push_back(r);
            area_points[r2id[r]][c2id[c]]++;
        }

        for (int rid = 0; rid <= H; rid++) {
            for (int cid = 0; cid <= W; cid++) {
                now[area_points[rid][cid]]++;
            }
        }
        for (int d = 1; d <= 1000; d++) cost += d * abs(target[d] - now[d]);
    }

    inline void area_change(int rid, int cid, int diff) {
        int& n = area_points[rid][cid];
        cost -= n * abs(target[n] - now[n]);
        now[n]--;
        cost += n * abs(target[n] - now[n]);
        n += diff;
        cost -= n * abs(target[n] - now[n]);
        now[n]++;
        cost += n * abs(target[n] - now[n]);
    }

    // 番号 rid の横線を下にずらす
    // 線上にある点が rid-1 に移る
    void move_down(int rid) {
        assert(rid >= 1 && rid <= H);
        assert(rpos[rid] != rpos[rid + 1]);
        int r = rpos[rid];
        for (int c : r2cs[r]) {
            int cid = c2id[c];
            area_change(rid, cid, -1);
            area_change(rid - 1, cid, 1);
        }
        r2id[r] = rid - 1;
        rpos[rid]++;
    }

    void move_down(int rid, int dist) {
        for (int d = 0; d < dist; d++) move_down(rid);
    }

    // 番号 rid の横線を上にずらす
    // 1 つ上にある点が rid に移る
    void move_up(int rid) {
        assert(rid >= 1 && rid <= H);
        assert(rpos[rid] != rpos[rid - 1]);
        int r = rpos[rid] - 1;
        for (int c : r2cs[r]) {
            int cid = c2id[c];
            area_change(rid - 1, cid, -1);
            area_change(rid, cid, 1);
        }
        r2id[r] = rid;
        rpos[rid]--;
    }

    void move_up(int rid, int dist) {
        for (int d = 0; d < dist; d++) move_up(rid);
    }

    void move_right(int cid) {
        assert(cid >= 1 && cid <= W);
        assert(cpos[cid] != cpos[cid + 1]);
        int c = cpos[cid];
        for (int r : c2rs[c]) {
            int rid = r2id[r];
            area_change(rid, cid, -1);
            area_change(rid, cid - 1, 1);
        }
        c2id[c] = cid - 1;
        cpos[cid]++;
    }

    void move_right(int cid, int dist) {
        for (int d = 0; d < dist; d++) move_right(cid);
    }

    void move_left(int cid) {
        assert(cid >= 1 && cid <= W);
        assert(cpos[cid] != cpos[cid - 1]);
        int c = cpos[cid] - 1;
        for (int r : c2rs[c]) {
            int rid = r2id[r];
            area_change(rid, cid - 1, -1);
            area_change(rid, cid, 1);
        }
        c2id[c] = cid;
        cpos[cid]--;
    }

    void move_left(int cid, int dist) {
        for (int d = 0; d < dist; d++) move_left(cid);
    }

    Trans move_random(Xorshift& rnd) {
        int id = -1, dir = -1, dist = -1;
        while (true) {
            dir = rnd.next_int(4);
            id = (dir & 1) ? (rnd.next_int(H) + 1) : (rnd.next_int(W) + 1);

            if (dir == 0) {
                if (cpos[id] == cpos[id + 1]) continue;
                dist = std::max(1, rnd.next_int(cpos[id + 1] - cpos[id]));
                break;
            }

            if (dir == 1) {
                if (rpos[id] == rpos[id - 1]) continue;
                dist = std::max(1, rnd.next_int(rpos[id] - rpos[id - 1]));
                break;
            }

            if (dir == 2) {
                if (cpos[id] == cpos[id - 1]) continue;
                dist = std::max(1, rnd.next_int(cpos[id] - cpos[id - 1]));
                break;
            }

            if (dir == 3) {
                if (rpos[id] == rpos[id + 1]) continue;
                dist = std::max(1, rnd.next_int(rpos[id + 1] - rpos[id]));
                break;
            }
        }

        chmin(dist, rnd.next_int(5) + 1);

        int prev_cost = cost;

        if (dir == 0) move_right(id, dist);
        if (dir == 1) move_up(id, dist);
        if (dir == 2) move_left(id, dist);
        if (dir == 3) move_down(id, dist);

        int diff = cost - prev_cost;

        return { id, dir, dist, diff };
    }

    void undo(const Trans& t) {
        if (t.dir == 0) move_left(t.id, t.dist);
        if (t.dir == 1) move_down(t.id, t.dist);
        if (t.dir == 2) move_right(t.id, t.dist);
        if (t.dir == 3) move_up(t.id, t.dist);
    }

    Output get_output() const {
        Output output;
        for (int rid = 1; rid <= H; rid++) {
            output.emplace_back(-10000, ys[rpos[rid]], 10000, ys[rpos[rid]] - 1);
        }
        for (int cid = 1; cid <= W; cid++) {
            output.emplace_back(xs[cpos[cid]], -10000, xs[cpos[cid]] - 1, 10000);
        }
        return output;
    }

};

int debug_count = 0;

struct State {

    const Input& input;

    int N;
    //int P;
    int H, W;
    int target[1001]; // 最大 1000 個
    int now[1001];
    int cost; // sum(abs(target[i]-now[i]))

    vector<int> rows; // 横線 [0, 20000]
    vector<int> cols; // 縦線 [0, 20000]
    // 0, 20000 は固定で、それ以外を動かす

    // 座標から行番号・列番号を逆引きできるようにする
    int y2rid[20001];
    int x2cid[20001];

    vector<vector<int>> area_points; // 区間に入っている点の個数

    // 座標 y 上にある番号 r の横線を下にずらす
    // y 上にある点が r-1 に移る
    // y+1 上にある点が境界に乗る
    void move_down(int rid) {
        assert(rid >= 1 && rid <= H);
        assert(rows[rid] + 1 != rows[rid + 1]);
        int y = rows[rid];
        for (int x : input.y2xs[y]) {
            int cid = x2cid[x];
            if (cid == -1) continue; // TODO: 境界用の添字を用意する
            int& n = area_points[rid - 1][cid];
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]--;
            if (n) cost += n * abs(target[n] - now[n]);
            n++;
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]++;
            if (n) cost += n * abs(target[n] - now[n]);
        }
        for (int x : input.y2xs[y + 1]) {
            int cid = x2cid[x];
            if (cid == -1) continue;
            int& n = area_points[rid][cid];
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]--;
            if (n) cost += n * abs(target[n] - now[n]);
            n--;
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]++;
            if (n) cost += n * abs(target[n] - now[n]);
        }
        y2rid[y] = rid - 1;
        y2rid[y + 1] = -1;
        rows[rid]++;
    }

    void move_down(int rid, int dist) {
        for (int d = 0; d < dist; d++) move_down(rid);
    }

    // 座標 y 上にある番号 r の横線を上にずらす
    // y 上にある点が r に移る
    // y-1 上にある点が境界に乗る
    void move_up(int rid) {
        assert(rid >= 1 && rid <= H);
        if (!(rows[rid] - 1 != rows[rid - 1])) {
            dump(debug_count);
        }
        assert(rows[rid] - 1 != rows[rid - 1]);
        int y = rows[rid];
        for (int x : input.y2xs[y]) {
            int cid = x2cid[x];
            if (cid == -1) continue; // TODO: 境界用の添字を用意する
            int& n = area_points[rid][cid];
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]--;
            if (n) cost += n * abs(target[n] - now[n]);
            n++;
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]++;
            if (n) cost += n * abs(target[n] - now[n]);
        }
        for (int x : input.y2xs[y - 1]) {
            int cid = x2cid[x];
            if (cid == -1) continue;
            int& n = area_points[rid - 1][cid];
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]--;
            if (n) cost += n * abs(target[n] - now[n]);
            n--;
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]++;
            if (n) cost += n * abs(target[n] - now[n]);
        }
        y2rid[y] = rid;
        y2rid[y - 1] = -1;
        rows[rid]--;
    }

    void move_up(int rid, int dist) {
        for (int d = 0; d < dist; d++) move_up(rid);
    }

    // 座標 x 上にある番号 c の横線を右にずらす
    // x 上にある点が c-1 に移る
    // x+1 上にある点が境界に乗る
    void move_right(int cid) {
        assert(cid >= 1 && cid <= W);
        assert(cols[cid] + 1 != cols[cid + 1]);
        int x = cols[cid];
        for (int y : input.x2ys[x]) {
            int rid = y2rid[y];
            if (rid == -1) continue; // TODO: 境界用の添字を用意する
            int& n = area_points[rid][cid - 1];
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]--;
            if (n) cost += n * abs(target[n] - now[n]);
            n++;
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]++;
            if (n) cost += n * abs(target[n] - now[n]);
        }
        for (int y : input.x2ys[x + 1]) {
            int rid = y2rid[y];
            if (rid == -1) continue;
            int& n = area_points[rid][cid];
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]--;
            if (n) cost += n * abs(target[n] - now[n]);
            n--;
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]++;
            if (n) cost += n * abs(target[n] - now[n]);
        }
        x2cid[x] = cid - 1;
        x2cid[x + 1] = -1;
        cols[cid]++;
    }

    void move_right(int cid, int dist) {
        for (int d = 0; d < dist; d++) move_right(cid);
    }

    // 座標 x 上にある番号 c の横線を右にずらす
    // x 上にある点が c に移る
    // x-1 上にある点が境界に乗る
    void move_left(int cid) {
        assert(cid >= 1 && cid <= W);
        assert(cols[cid] - 1 != cols[cid - 1]);
        int x = cols[cid];
        for (int y : input.x2ys[x]) {
            int rid = y2rid[y];
            if (rid == -1) continue; // TODO: 境界用の添字を用意する
            int& n = area_points[rid][cid];
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]--;
            if (n) cost += n * abs(target[n] - now[n]);
            n++;
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]++;
            if (n) cost += n * abs(target[n] - now[n]);
        }
        for (int y : input.x2ys[x - 1]) {
            int rid = y2rid[y];
            if (rid == -1) continue;
            int& n = area_points[rid][cid - 1];
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]--;
            if (n) cost += n * abs(target[n] - now[n]);
            n--;
            if (n) cost -= n * abs(target[n] - now[n]);
            now[n]++;
            if (n) cost += n * abs(target[n] - now[n]);
        }
        x2cid[x] = cid;
        x2cid[x - 1] = -1;
        cols[cid]--;
    }

    void move_left(int cid, int dist) {
        for (int d = 0; d < dist; d++) move_left(cid);
    }

    Trans move_random(Xorshift& rnd, int cap = 10) {
        int id = -1, dir = -1, dist = -1;
        while (true) {
            dir = rnd.next_int(4);
            id = (dir & 1) ? (rnd.next_int(H) + 1) : (rnd.next_int(W) + 1);

            if (dir == 0) {
                if (cols[id] + 1 == cols[id + 1]) continue;
                dist = std::max(1, rnd.next_int(cols[id + 1] - cols[id]));
                break;
            }

            if (dir == 1) {
                if (rows[id] - 1 == rows[id - 1]) continue;
                dist = std::max(1, rnd.next_int(rows[id] - rows[id - 1]));
                break;
            }

            if (dir == 2) {
                if (cols[id] - 1 == cols[id - 1]) continue;
                dist = std::max(1, rnd.next_int(cols[id] - cols[id - 1]));
                break;
            }

            if (dir == 3) {
                if (rows[id] + 1 == rows[id + 1]) continue;
                dist = std::max(1, rnd.next_int(rows[id + 1] - rows[id]));
                break;
            }
        }

        chmin(dist, cap);

        int prev_cost = cost;

        if (dir == 0) move_right(id, dist);
        if (dir == 1) move_up(id, dist);
        if (dir == 2) move_left(id, dist);
        if (dir == 3) move_down(id, dist);

        int diff = cost - prev_cost;

        return { id, dir, dist, diff };
    }

    void undo(const Trans& t) {
        if (t.dir == 0) move_left(t.id, t.dist);
        if (t.dir == 1) move_down(t.id, t.dist);
        if (t.dir == 2) move_right(t.id, t.dist);
        if (t.dir == 3) move_up(t.id, t.dist);
    }

    State(const Input& input, int H, int W) : input(input), N(input.N), H(H), W(W) {
        memset(target, 0, sizeof(int) * 1001);
        for (int d = 0; d < 10; d++) target[d + 1] = input.a[d];
        memset(now, 0, sizeof(int) * 1001);
        cost = 0;
        rows.resize(H + 2);
        cols.resize(W + 2);
        // [0, 20000] でだいたい等間隔に
        int row_interval = 20000 / (H + 1);
        int col_interval = 20000 / (W + 1);
        rows[0] = 0; rows[H + 1] = 20000;
        cols[0] = 0; cols[W + 1] = 20000;
        for (int i = 1; i <= H; i++) {
            int y = row_interval * i;
            rows[i] = y;
        }
        for (int j = 1; j <= W; j++) {
            int x = col_interval * j;
            cols[j] = x;
        }
        memset(y2rid, -1, sizeof(int) * 20001);
        memset(x2cid, -1, sizeof(int) * 20001);
        for (int rid = 0; rid + 1 < rows.size(); rid++) {
            int y1 = rows[rid] + 1, y2 = rows[rid + 1] - 1;
            for (int y = y1; y <= y2; y++) {
                y2rid[y] = rid;
            }
        }
        for (int cid = 0; cid + 1 < cols.size(); cid++) {
            int x1 = cols[cid] + 1, x2 = cols[cid + 1] - 1;
            for (int x = x1; x <= x2; x++) {
                x2cid[x] = cid;
            }
        }
        area_points.resize(H + 1, vector<int>(W + 1, 0));
        for (const auto& [x, y] : input.xys) {
            if (y2rid[y] == -1 || x2cid[x] == -1) continue;
            area_points[y2rid[y]][x2cid[x]]++;
        }
        for (int rid = 0; rid <= H; rid++) {
            for (int cid = 0; cid <= W; cid++) {
                now[area_points[rid][cid]]++;
            }
        }
        for (int d = 1; d <= 1000; d++) cost += d * abs(target[d] - now[d]);
    }

    Output get_output() const {
        Output output;
        for (int rid = 1; rid <= H; rid++) {
            output.emplace_back(0, rows[rid], 20000, rows[rid]);
        }
        for (int cid = 1; cid <= W; cid++) {
            output.emplace_back(cols[cid], 0, cols[cid], 20000);
        }
        return output;
    }

};

int compute_score(const Input& input, const Output& output) {
    vector<vector<int>> pieces;
    {
        vector<int> ps(input.N);
        std::iota(ps.begin(), ps.end(), 0);
        pieces.push_back(ps);
    }
    for (const auto& [px, py, qx, qy] : output) {
        vector<vector<int>> new_pieces;
        for (const auto& piece : pieces) {
            vector<int> left, right;
            for (int j : piece) {
                auto [x, y] = input.xys[j];
                ll side = ll(qx - px) * ll(y - py) - ll(qy - py) * ll(x - px);
                if (side > 0) left.push_back(j);
                else if (side < 0) right.push_back(j);
            }
            if (left.size()) new_pieces.push_back(left);
            if (right.size()) new_pieces.push_back(right);
        }
        pieces = new_pieces;
    }
    vector<int> b(10);
    for (const auto& piece : pieces) {
        if (piece.size() <= 10) {
            b[piece.size() - 1]++;
        }
    }
    int num = 0, den = 0;
    for (int d = 0; d < 10; d++) {
        num += std::min(input.a[d], b[d]);
        den += input.a[d];
    }
    int score = (int)round(1e6 * num / den);
    return score;
}

int compute_score(const Input2& input, const Output& output) {
    vector<vector<int>> pieces;
    {
        vector<int> ps(input.N);
        std::iota(ps.begin(), ps.end(), 0);
        pieces.push_back(ps);
    }
    for (const auto& [px, py, qx, qy] : output) {
        vector<vector<int>> new_pieces;
        for (const auto& piece : pieces) {
            vector<int> left, right;
            for (int j : piece) {
                auto [x, y] = input.xys[j];
                ll side = ll(qx - px) * ll(y - py) - ll(qy - py) * ll(x - px);
                if (side > 0) left.push_back(j);
                else if (side < 0) right.push_back(j);
            }
            if (left.size()) new_pieces.push_back(left);
            if (right.size()) new_pieces.push_back(right);
        }
        pieces = new_pieces;
    }
    vector<int> b(10);
    for (const auto& piece : pieces) {
        if (piece.size() <= 10) {
            b[piece.size() - 1]++;
        }
    }
    int num = 0, den = 0;
    for (int d = 0; d < 10; d++) {
        num += std::min(input.a[d], b[d]);
        den += input.a[d];
    }
    int score = (int)round(1e6 * num / den);
    return score;
}

void test() {
    std::ifstream ifs(R"(tools\in\0000.txt)");
    std::istream& in = ifs;
    Input input(in);

    string ans = R"(30
-1812 -1984 -9663 5131
-2586 7859 7423 -4798
193 8457 -300 2680
-149 -9041 6425 84
3329 -5601 -2055 -5156
4701 3753 -2852 -2578
-1417 -9909 -5593 -5639
-7013 2309 -7622 6652
-5296 -5647 -9646 -1031
-1275 8332 6134 1727
-6996 2724 933 4067
1134 3022 -9377 2501
6487 -7016 -6730 4775
9141 -7928 7794 -4177
-5164 -3440 -2341 -6820
-5940 2165 -1255 6089
1691 -1616 347 -8205
-7966 -6024 -2883 5136
6899 3888 1191 6179
-762 7446 2182 2919
-6843 1720 -8636 -9995
-2997 -6378 4359 -2044
-7094 2392 2381 558
-3241 7718 -2179 -1519
-358 247 -547 -5282
-376 -1900 7187 3663
7470 1828 4310 -103
-2289 7865 8712 -1962
-6942 -5180 -1909 326
3266 3617 3585 1090
)";

    std::istringstream iss(ans);
    Output output;
    int k;
    iss >> k;
    for (int i = 0; i < k; i++) {
        int px, py, qx, qy;
        iss >> px >> py >> qx >> qy;
        output.emplace_back(px, py, qx, qy);
    }

    dump(output);
    dump(compute_score(input, output)); // 244353
}

void output(std::ostream& out, const Output& output) {
    out << output.size() << '\n';
    for (const auto& [px, py, qx, qy] : output) {
        out << px << ' ' << py << ' ' << qx << ' ' << qy << '\n';
    }
}

struct Result {
    Output output;
    int score;
    int loop;
    int cost;
};

Result solve(const Input& input) {

    Timer timer;

    int H = 8, W = 92;
    State state(input, H, W);

    auto get_temp = [](double startTemp, double endTemp, double t, double T) {
        return endTemp + (startTemp - endTemp) * (T - t) / T;
    };

    int loop = 0;
    double start_time = timer.elapsed_ms(), now_time, end_time = 2900;
    while ((now_time = timer.elapsed_ms()) < end_time) {

        auto trans = state.move_random(rnd);

        double temp = get_temp(5.0, 0.0, now_time - start_time, end_time - start_time);
        double prob = exp(-trans.diff / temp);

        if (rnd.next_double() > prob) {
            state.undo(trans);
        }

        loop++;
    }

    Result res;
    res.output = state.get_output();
    res.score = compute_score(input, res.output);
    res.loop = loop;
    res.cost = state.cost;

    return res;
}

Result solve2(const Input2& input) {

    Timer timer;

    int H = 8, W = 92;
    State2 state(input, H, W);

    auto get_temp = [](double startTemp, double endTemp, double t, double T) {
        return endTemp + (startTemp - endTemp) * (T - t) / T;
    };

    int loop = 0;
    double start_time = timer.elapsed_ms(), now_time, end_time = 2900;
    while ((now_time = timer.elapsed_ms()) < end_time) {

        auto trans = state.move_random(rnd);

        double temp = get_temp(10.0, 0.0, now_time - start_time, end_time - start_time);
        double prob = exp(-trans.diff / temp);

        if (rnd.next_double() > prob) {
            state.undo(trans);
        }

        //if (!(loop & 0x3FFFFF)) {
        //    dump(loop, state.cost);
        //}

        loop++;
    }
    //dump(state.cost);

    Result res;
    res.output = state.get_output();
    res.score = 1000000 - compute_score(input, res.output);
    res.loop = loop;
    res.cost = state.cost;
    //dump(res.score, res.score, res.cost);

    return res;
}

#ifdef _MSC_VER
void batch_test(int seed_begin = 0, int num_seed = 100) {

    constexpr int batch_size = 8;
    int seed_end = seed_begin + num_seed;

    vector<int> scores(num_seed);
    concurrency::critical_section mtx;
    for (int batch_begin = seed_begin; batch_begin < seed_end; batch_begin += batch_size) {
        int batch_end = std::min(batch_begin + batch_size, seed_end);
        concurrency::parallel_for(batch_begin, batch_end, [&mtx, &scores](int seed) {
            std::ifstream ifs(format("tools/in/%04d.txt", seed));
            std::istream& in = ifs;
            std::ofstream ofs(format("tools/out/%04d.txt", seed));
            std::ostream& out = ofs;

            Input2 input(in);
            auto res = solve2(input);
            output(out, res.output);

            {
                mtx.lock();
                scores[seed] = res.score;
                cerr << seed << ": " << res.score << '\n';
                mtx.unlock();
            }
        });
    }

    dump(std::accumulate(scores.begin(), scores.end(), 0));
}
#endif


int main(int argc, char** argv) {

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#ifdef _MSC_VER
    std::ifstream ifs(R"(tools\in\0000.txt)");
    std::ofstream ofs(R"(tools\out\0000.txt)");
    std::istream& in = ifs;
    std::ostream& out = ofs;
#else
    std::istream& in = cin;
    std::ostream& out = cout;
#endif

#if 1
    Input2 input2(in);

    auto res = solve2(input2);

    dump(res.score, res.loop, res.cost);

    output(out, res.output);
#else
    batch_test();
#endif

    return 0;
}