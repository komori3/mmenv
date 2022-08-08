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

using namespace std;


#define BATCH_TEST
#ifdef BATCH_TEST
#undef ENABLE_DUMP
#endif


int N, C, P;
static const int dx[] = { 1,0,-1,0 };
static const int dy[] = { 0,1,0,-1 };

int gridv[32][32];
int gridc[32][32];
int gridi[32][32];

double get_time() {
    auto now = chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    auto epoch_micros = chrono::duration_cast<chrono::microseconds>(epoch);
    return 1e-6 * epoch_micros.count();
}

struct SPipe
{
    vector<int> x;
    vector<int> y;
};

vector<int> sx, sy, sval, scol;

struct SSol
{
    int score;
    vector<SPipe> pipe;
    char used[32][32];
};

SSol osol, bsol;
double dStartTime;

void clearSol(SSol& sol)
{
    sol.score = 0;
    sol.pipe.clear();
    memset(sol.used, 0, sizeof(sol.used));
}

void outputSol(std::ostream& out, SSol& sol)
{
    out << sol.pipe.size() << endl;
    for (auto p : sol.pipe)
    {
        out << p.x.size() << endl;
        // cerr << p.x.size() << endl;
        for (int i = 0; i < p.x.size(); i++)
        {
            out << p.y[i] << " " << p.x[i] << endl;
            // cerr << p.y[i] << "," << p.x[i] << " ";
        }
        // cerr << endl;
    }
    out.flush();
}

void addPipe(SSol& sol, vector<int>& px, vector<int>& py)
{
    SPipe p;
    p.x = px;
    p.y = py;
    sol.pipe.push_back(p);
    for (int i = 0; i < px.size(); i++)
    {
        if (sol.used[px[i]][py[i]] > 0)
            sol.score -= P * sol.used[px[i]][py[i]];
        sol.used[px[i]][py[i]]++;
    }
    sol.score += gridv[px[0]][py[0]] * gridv[px[px.size() - 1]][py[py.size() - 1]];
}

void calc_score(SSol& sol)
{
}

vector< pair<int, int> > findAllPairs(SSol& sol)
{
    vector< pair<int, int> > prs;
    for (int i1 = 0; i1 < sx.size(); i1++) if (!sol.used[sx[i1]][sy[i1]])
        for (int i2 = i1 + 1; i2 < sx.size(); i2++) if (!sol.used[sx[i2]][sy[i2]] && scol[i1] == scol[i2])
        {
            prs.push_back(make_pair(i1, i2));
        }
    return prs;
}

bool findPath(SSol& sol, int i1, int i2)
{
    list<int> qx, qy, qcost, qcross;
    int gval = sval[i1] * sval[i2];
    int cost[32][32];
    int from[32][32];
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
        {
            from[x][y] = -1;
            cost[x][y] = 1 << 20;
        }
    cost[sx[i1]][sy[i1]] = 0;
    qx.push_back(sx[i1]);
    qy.push_back(sy[i1]);
    qcost.push_back(0);
    qcross.push_back(0);
    while (!qx.empty())
    {
        int px = qx.front(); qx.pop_front();
        int py = qy.front(); qy.pop_front();
        int c = qcost.front(); qcost.pop_front();
        int cross = qcross.front(); qcross.pop_front();
        if (c == cost[px][py])
            for (int d = 0; d < 4; d++)
            {
                int nx = px + dx[d];
                int ny = py + dy[d];
                if (nx >= 0 && nx < N && ny >= 0 && ny < N)
                {
                    if (nx == sx[i2] && ny == sy[i2])
                    {
                        if (c + 1 < cost[nx][ny])
                        {
                            cost[nx][ny] = c + 1;
                            from[nx][ny] = d;
                        }
                    }
                    else
                    {
                        int nc = c + 1;
                        if (sol.used[nx][ny] > 0)
                            nc += sol.used[nx][ny];
                        int ncross = cross;
                        if (sol.used[nx][ny] > 0)
                            ncross += sol.used[nx][ny];
                        if (gridc[nx][ny] == 0 && nc < cost[nx][ny] && gval - ncross * P>0)
                        {
                            qx.push_back(nx);
                            qy.push_back(ny);
                            qcost.push_back(nc);
                            qcross.push_back(ncross);
                            cost[nx][ny] = nc;
                            from[nx][ny] = d;
                        }
                    }

                }

            }
    }
    if (from[sx[i2]][sy[i2]] >= 0)
    {
        // reached the goal
        int nx = sx[i2];
        int ny = sy[i2];
        vector<int> rx, ry;
        rx.push_back(nx);
        ry.push_back(ny);
        int dd = from[nx][ny];
        do
        {
            nx -= dx[dd];
            ny -= dy[dd];
            rx.push_back(nx);
            ry.push_back(ny);
            dd = from[nx][ny];
        } while (dd >= 0);
        reverse(rx.begin(), rx.end());
        reverse(ry.begin(), ry.end());
        addPipe(sol, rx, ry);
        return true;
    }
    return false;

}

bool randomPick(SSol& sol)
{
    vector< pair<int, int> > prs = findAllPairs(sol);
    // shuffle
    for (int i = 0; i < prs.size(); i++)
    {
        int i1 = rand() % prs.size();
        swap(prs[i], prs[i1]);
    }
    for (int i = 0; i < prs.size(); i++)
    {
        if (findPath(sol, prs[i].first, prs[i].second))
        {
            return true;
        }
    }
    return false;
}

void doOrder(SSol& sol, vector< pair<int, int> >& prs)
{
    for (int i = 0; i < prs.size(); i++)
    {
        int i1 = prs[i].first;
        int i2 = prs[i].second;
        if (sol.used[sx[i1]][sy[i1]]) continue;
        if (sol.used[sx[i2]][sy[i2]]) continue;
        findPath(sol, i1, i2);
    }
}



int main(int argc, char** argv) {

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#if 0
    std::ifstream ifs(R"(C:\Users\komori3\OneDrive\dev\compro\heuristic\tasks\MM137\tester\in\2.in)");
    std::ofstream ofs(R"(C:\Users\komori3\OneDrive\dev\compro\heuristic\tasks\MM137\tester\out\2.out)");
    std::istream& in = ifs;
    std::ostream& out = ofs;
#else
    std::istream& in = cin;
    std::ostream& out = cout;
#endif


    dStartTime = get_time();
    srand(19790206);

    in >> N >> C >> P;
    cerr << "N = " << N << " C = " << C << " P = " << P << endl;
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
        {
            int col, val;
            in >> val >> col;
            gridc[c][r] = col;
            gridv[c][r] = val;
            gridi[c][r] = -1;
            if (col > 0)
            {
                gridi[c][r] = sx.size();
                sx.push_back(c);
                sy.push_back(r);
                sval.push_back(val);
                scol.push_back(col);
            }
        }
    // for (int y=0;y<N;y++)
    // {
    //   for (int x=0;x<N;x++)
    //   {
    //     cerr << gridc[x][y] << " ";
    //   }
    //   cerr << endl;
    // }

    clearSol(osol);
    clearSol(bsol);
    int itr = 0;

    vector< pair<int, int> > order = findAllPairs(osol);
    vector< pair<int, int> > srt;
    for (int i1 = 0; i1 < order.size(); i1++)
    {
        int dst = abs(sx[order[i1].first] - sx[order[i1].second]) + abs(sy[order[i1].first] - sy[order[i1].second]);
        srt.push_back(make_pair(-sval[order[i1].first] * sval[order[i1].second] * (N - dst), i1));
    }
    sort(srt.begin(), srt.end());
    vector< pair<int, int> > border = order;
    for (int i1 = 0; i1 < order.size(); i1++)
        border[i1] = order[srt[i1].second];

    while (get_time() - dStartTime < 9.0)
    {
        itr++;
        SSol sol = osol;
        order = border;
        //for (int i=0;i<sx.size();i++)
        if (itr != 1)
            for (int i = 0; i < 1 + sx.size() / 10; i++)
            {
                int i1 = rand() % order.size();
                int i2 = rand() % order.size();
                swap(order[i1], order[i2]);
            }
        doOrder(sol, order);
        //while (randomPick(sol)) { }

        if (sol.score > bsol.score)
        {
            bsol = sol;
            border = order;
            cerr << sol.score << endl;
        }
        // break;
    }

    outputSol(out, bsol);
    cerr << "itr  = " << itr << endl;
    cerr << "Time = " << get_time() - dStartTime << endl;
    cerr << "sc   = " << bsol.score << endl;


    return 0;
}