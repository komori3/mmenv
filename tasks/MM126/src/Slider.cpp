//#define NDEBUG
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include "bits/stdc++.h"
#include <memory.h>
#include <unordered_map>
#include <unordered_set>
#include <random>
#ifdef _MSC_VER
#include <ppl.h>
#include <concurrent_vector.h>
//#include <boost/multiprecision/cpp_dec_float.hpp>
//#include <boost/multiprecision/cpp_int.hpp>
//#include <boost/rational.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#else
//#include <omp.h>
#endif

/* type */
using uint = unsigned; using ll = long long; using ull = unsigned long long; using pii = std::pair<int, int>; using pll = std::pair<ll, ll>;
/* io */
template <typename _KTy, typename _Ty> std::ostream& operator << (std::ostream& o, const std::pair<_KTy, _Ty>& m) { o << "[" << m.first << ", " << m.second << "]"; return o; }
template <typename _KTy, typename _Ty> std::ostream& operator << (std::ostream& o, const std::map<_KTy, _Ty>& m) { if (m.empty()) { o << "[]"; return o; } o << "[" << *m.begin(); for (auto itr = ++m.begin(); itr != m.end(); itr++) { o << ", " << *itr; } o << "]"; return o; }
template <typename _KTy, typename _Ty> std::ostream& operator << (std::ostream& o, const std::unordered_map<_KTy, _Ty>& m) { if (m.empty()) { o << "[]"; return o; } o << "[" << *m.begin(); for (auto itr = ++m.begin(); itr != m.end(); itr++) { o << ", " << *itr; } o << "]"; return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::vector<_Ty>& v) { if (v.empty()) { o << "[]"; return o; } o << "[" << v.front(); for (auto itr = ++v.begin(); itr != v.end(); itr++) { o << ", " << *itr; } o << "]"; return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::deque<_Ty>& v) { if (v.empty()) { o << "[]"; return o; } o << "[" << v.front(); for (auto itr = ++v.begin(); itr != v.end(); itr++) { o << ", " << *itr; } o << "]"; return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::set<_Ty>& s) { if (s.empty()) { o << "[]"; return o; } o << "[" << *(s.begin()); for (auto itr = ++s.begin(); itr != s.end(); itr++) { o << ", " << *itr; } o << "]"; return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::unordered_set<_Ty>& s) { if (s.empty()) { o << "[]"; return o; } o << "[" << *(s.begin()); for (auto itr = ++s.begin(); itr != s.end(); itr++) { o << ", " << *itr; }	o << "]"; return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::stack<_Ty>& s) { if (s.empty()) { o << "[]"; return o; } std::stack<_Ty> t(s); o << "[" << t.top(); t.pop(); while (!t.empty()) { o << ", " << t.top(); t.pop(); } o << "]";	return o; }
template <typename _Ty> std::ostream& operator << (std::ostream& o, const std::list<_Ty>& l) { if (l.empty()) { o << "[]"; return o; } o << "[" << l.front(); for (auto itr = ++l.begin(); itr != l.end(); ++itr) { o << ", " << *itr; } o << "]"; return o; }
template <typename _KTy, typename _Ty> std::istream& operator >> (std::istream& is, std::pair<_KTy, _Ty>& m) { is >> m.first >> m.second; return is; }
template <typename _Ty> std::istream& operator >> (std::istream& is, std::vector<_Ty>& v) { for (size_t t = 0; t < v.size(); t++) is >> v[t]; return is; }
template <typename _Ty> std::istream& operator >> (std::istream& is, std::deque<_Ty>& v) { for (size_t t = 0; t < v.size(); t++) is >> v[t]; return is; }
template<typename T, size_t N> std::ostream& operator<<(std::ostream& o, const std::array<T, N>& a) { if (a.empty()) { o << "[]"; return o; } o << "[" << a[0]; for (int i = 1; i < N; i++) o << ", " << a[i]; o << "]"; return o; }
namespace aux { // print tuple
  template<typename Ty, unsigned N, unsigned L> struct tp { static void print(std::ostream& os, const Ty& v) { os << std::get<N>(v) << ", "; tp<Ty, N + 1, L>::print(os, v); } };
  template<typename Ty, unsigned N> struct tp<Ty, N, N> { static void print(std::ostream& os, const Ty& v) { os << std::get<N>(v); } };
}
template<typename... Tys> std::ostream& operator<<(std::ostream& os, const std::tuple<Tys...>& t) { os << "["; aux::tp<std::tuple<Tys...>, 0, sizeof...(Tys) - 1>::print(os, t); os << "]"; return os; }
/* fill */
template<typename A, size_t N, typename T>
void Fill(A(&array)[N], const T& val) {
  std::fill((T*)array, (T*)(array + N), val);
}
/* format */
template <typename ... Args>
std::string format(const std::string& fmt, Args ... args) {
  size_t len = std::snprintf(nullptr, 0, fmt.c_str(), args ...);
  std::vector<char> buf(len + 1);
  std::snprintf(&buf[0], len + 1, fmt.c_str(), args ...);
  return std::string(&buf[0], &buf[0] + len);
}
/* dump */
#define ENABLE_DUMP
#ifdef ENABLE_DUMP
#define DUMPOUT std::cerr
#define dump(...) do{DUMPOUT<<"  ";DUMPOUT<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPOUT<<"    ";dump_func(__VA_ARGS__);}while(0);
void dump_func() { DUMPOUT << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPOUT << head; if (sizeof...(Tail) == 0) { DUMPOUT << " "; } else { DUMPOUT << ", "; } dump_func(std::move(tail)...); }
#else
#define dump(...) void(0);
#endif
/* timer */
class Timer {
public:
  double t = 0;
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
  //void measure() { t = time() - t; }
  void reset() { t = time(); }
  double elapsedMs() { return (time() - t) * 1000.0; }
} timer;
/* rand */
struct Xorshift {
  uint64_t x = 88172645463325252LL;
  unsigned next_int() {
    x = x ^ (x << 7);
    return x = x ^ (x >> 9);
  }
  unsigned next_int(unsigned mod) {
    x = x ^ (x << 7);
    x = x ^ (x >> 9);
    return x % mod;
  }
  unsigned next_int(unsigned l, unsigned r) {
    x = x ^ (x << 7);
    x = x ^ (x >> 9);
    return x % (r - l + 1) + l;
  }
  double next_double() {
    return double(next_int()) / UINT_MAX;
  }
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

using namespace std;

constexpr int dr[] = { 0, -1, 0, 1 };
constexpr int dc[] = { 1, 0, -1, 0 };
constexpr char dtoc[] = "RULD";
char ctod[256];

constexpr int HOLE = -1;
constexpr int WALL = -2;
constexpr int SPACE = 0;
int N, C, H;
vector<vector<int>> grid;
vector<pii> holes;
vector<vector<bool>> hole_visible;
vector<vector<int>> hole_distance;

inline bool is_inside(int r, int c) {
  return 0 <= r && r < N && 0 <= c && c < N;
}

void init(istream& in) {
  ctod['R'] = 0; ctod['U'] = 1; ctod['L'] = 2; ctod['D'] = 3;
  in >> N >> C >> H;
  grid.resize(N, vector<int>(N));
  // hole
  in >> grid;
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < N; c++) {
      if (grid[r][c] == HOLE) {
        holes.emplace_back(r, c);
      }
    }
  }
  // distance
  hole_distance.resize(N, vector<int>(N));
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < N; c++) {
      if (grid[r][c] == HOLE) {
        hole_distance[r][c] = 0;
      }
      else {
        int min_dist = INT_MAX;
        for (const pii& h : holes) {
          int hr = h.first, hc = h.second;
          min_dist = min(min_dist, abs(hr - r) + abs(hc - c));
        }
        hole_distance[r][c] = min_dist;
      }
    }
  }
  // visible
  hole_visible.resize(N, vector<bool>(N, false));
  for (const pii& h : holes) {
    int hr = h.first, hc = h.second;
    for (int d = 0; d < 4; d++) {
      int r = hr + dr[d], c = hc + dc[d];
      while (is_inside(r, c)) {
        hole_visible[r][c] = true;
        r += dr[d]; c += dc[d];
      }
    }
  }
}

struct Cmd {
  int val;
  int pr, pc;
  int r, c;
  int d;
  bool slide;
  int diff;
  Cmd() {}
  Cmd(int val, int pr, int pc, int r, int c, int d, bool slide, int diff)
    : val(val), pr(pr), pc(pc), r(r), c(c), d(d), slide(slide), diff(diff) {}
  std::string str() const {
    return format("Cmd [val=%d, pr=%d, pc=%d, r=%d, c=%d, d=%d, slide=%d, diff=%d]", val, pr, pc, r, c, d, slide, diff);
  }
  void output(ostream& out) const {
    out << format("%d %d %c %c", pr, pc, slide ? 'S' : 'M', dtoc[d]);
  }
  friend std::ostream& operator<<(std::ostream& o, const Cmd& obj) {
    o << obj.str();
    return o;
  }
  friend std::ostream& operator<<(std::ostream& o, const shared_ptr<Cmd>& obj) {
    o << obj->str();
    return o;
  }
};

int debug_count = 0;

struct State {
  vector<vector<int>> grid;

  int mult;
  int raw_score;
  int distance_cost;
  vector<Cmd> cmds;

  bool verbose;

  static State create(bool verbose = false) {
    State s;
    s.raw_score = 0;
    s.grid = ::grid;
    s.mult = N * N;
    s.verbose = verbose;
    s.distance_cost = 0;
    for (int r = 0; r < N; r++) {
      for (int c = 0; c < N; c++) {
        if (s.grid[r][c] <= 0) continue;
        s.distance_cost += hole_distance[r][c] * (s.grid[r][c] - 1);
      }
    }
    return s;
  }

  bool can_move(int r, int c, int d) const {
    if (grid[r][c] <= 0) return false;
    int nr = r + dr[d], nc = c + dc[d];
    return is_inside(nr, nc) && grid[nr][nc] <= 0;
  }

  int _do_move(int r, int c, int d) {
    assert(grid[r][c] > 0);
    int val = grid[r][c];
    distance_cost -= hole_distance[r][c] * (val - 1);
    int nr = r + dr[d], nc = c + dc[d];
    if (grid[nr][nc] == HOLE) {
      int diff = mult * (val - 1);
      raw_score += diff;
      grid[r][c] = 0;
      return diff;
    }
    grid[nr][nc] = val;
    grid[r][c] = 0;
    distance_cost += hole_distance[nr][nc] * (val - 1);

    return 0;
  }

  void do_move(int r, int c, int d) {
    int val = grid[r][c];
    int diff = _do_move(r, c, d);
    cmds.emplace_back(val, r, c, r + dr[d], c + dc[d], d, false, diff);
    mult--;
  }

  void do_slide(int r, int c, int d) {
    int pr = r, pc = c, val = grid[r][c], diff = 0;
    while (can_move(r, c, d)) {
      if ((diff = _do_move(r, c, d))) {
        r += dr[d]; c += dc[d];
        break;
      }
      r += dr[d]; c += dc[d];
    }
    cmds.emplace_back(val, pr, pc, r, c, d, true, diff);
    mult--;
  }

  void do_cmd(const Cmd& cmd) {
    if (cmd.slide) do_slide(cmd.pr, cmd.pc, cmd.d);
    else do_move(cmd.pr, cmd.pc, cmd.d);
    assert(grid[cmd.pr][cmd.pc] == SPACE);
    if (verbose) print_stat(cerr);
  }

  tuple<int, int, int> detect(int r, int c, int d) const {
    r += dr[d]; c += dc[d];
    while (is_inside(r, c) && grid[r][c] == SPACE) {
      r += dr[d]; c += dc[d];
    }
    int val = is_inside(r, c) ? grid[r][c] : WALL;
    return { r, c, val };
  }

  pair<bool, Cmd> step_hole_in_one() const {
    // 最高効率の 1 ステップ操作を返す
    // TODO: 同点・黒を穴に入れる場合の評価
    if (mult == 0) return { false, Cmd() };
    int best_diff = -1;
    Cmd best_cmd;
    for (int r = 0; r < N; r++) {
      for (int c = 0; c < N; c++) {
        if (grid[r][c] != HOLE) continue;
        for (int d = 0; d < 4; d++) {
          int br, bc, bval;
          std::tie(br, bc, bval) = detect(r, c, d);
          int diff = mult * (bval - 1);
          if (best_diff < diff) {
            best_diff = diff;
            best_cmd = Cmd(bval, br, bc, r, c, (d + 2) & 3, true, diff);
          }
        }
      }
    }
    if (best_diff == -1) return { false, Cmd() };
    return { true, best_cmd };
  }

  void solve_hole_in_one() {
    while (true) {
      bool ok; Cmd cmd; std::tie(ok, cmd) = step_hole_in_one();
      if (!ok) break;
      do_cmd(cmd);
    }
  }

  pair<bool, Cmd> step_greedy() {
    if (mult == 0) return { false, Cmd() };
    int best_score = INT_MIN;
    Cmd best_cmd;
    for (int r = 0; r < N; r++) {
      for (int c = 0; c < N; c++) {
        for (int d = 0; d < 4; d++) {
          if (!can_move(r, c, d)) continue;
          for (int s = 0; s < 2; s++) {
            if (s) do_slide(r, c, d);
            else do_move(r, c, d);
            int score = raw_score - distance_cost;
            if (best_score < score) {
              best_score = score;
              best_cmd = cmds.back();
            }
            undo();
          }
        }
      }
    }
    if (best_score == INT_MIN) return { false, Cmd() };
    return { true, best_cmd };
  }

  void solve_greedy() {
    while (true) {
      bool ok; Cmd cmd; std::tie(ok, cmd) = step_greedy();
      if (!ok) break;
      do_cmd(cmd);
    }
  }

  void undo() {
    Cmd cmd = cmds.back(); cmds.pop_back();
    assert(grid[cmd.pr][cmd.pc] == SPACE);
    mult++;
    grid[cmd.pr][cmd.pc] = cmd.val;
    if (grid[cmd.r][cmd.c] == HOLE) {
      raw_score -= cmd.diff;
    }
    else {
      grid[cmd.r][cmd.c] = SPACE;
      distance_cost -= hole_distance[cmd.r][cmd.c] * (cmd.val - 1);
    }
    distance_cost += hole_distance[cmd.pr][cmd.pc] * (cmd.val - 1);
  }

  void output(ostream& out) const {
    out << cmds.size() << '\n';
    for (const Cmd& cmd : cmds) {
      cmd.output(out);
      out << '\n';
    }
    out.flush();
  }

  void print_stat(ostream& out) const {
    // score, mult, black, colored
    int black = 0, colored = 0;
    for (int r = 0; r < N; r++) {
      for (int c = 0; c < N; c++) {
        if (grid[r][c] == 1) black++;
        else if (grid[r][c] > 1) colored++;
      }
    }
    out << format("score:%7d, mult:%4d, black:%4d, colored:%4d", raw_score, mult, black, colored) << '\n';
  }

#ifdef _MSC_VER
  cv::Scalar get_color(int val) const {
    int g = (int)round(val * 255.0 / 9.0);
    return cv::Scalar(255 - g, g, 0);
  }
  void vis(int delay = 0) const {
    int grid_size = 30;
    cv::Mat_<cv::Vec3b> img(grid_size * N, grid_size * N, cv::Vec3b(255, 255, 255));
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        cv::Rect roi(j * grid_size, i * grid_size, grid_size, grid_size);
        cv::Mat_<cv::Vec3b> img_roi = img(roi);
        if (grid[i][j] == HOLE) {
          cv::circle(img_roi, cv::Point(15, 15), 10, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA);
        }
        else if (grid[i][j] == 1) {
          cv::rectangle(img_roi, cv::Rect(0, 0, grid_size, grid_size), cv::Scalar(0, 0, 0), cv::FILLED);
        }
        else if (grid[i][j] > 1) {
          cv::rectangle(img_roi, cv::Rect(0, 0, grid_size, grid_size), get_color(grid[i][j]), cv::FILLED);
          cv::putText(img_roi, to_string(grid[i][j]), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
      }
    }
    for (int i = 1; i < N; i++) cv::line(img, cv::Point(0, grid_size * i), cv::Point(grid_size * N, grid_size * i), cv::Scalar(0, 0, 0), 1);
    for (int j = 1; j < N; j++) cv::line(img, cv::Point(grid_size * j, 0), cv::Point(grid_size * j, grid_size * N), cv::Scalar(0, 0, 0), 1);
    cv::imshow("img", img);
    cv::waitKey(delay);
  }
#endif
};

//#define CONSOLE_MODE
int main() {
#ifdef CONSOLE_MODE
  ifstream ifs("C:\\dev\\TCMM\\problems\\MM126\\in\\42.in");
  istream& in = ifs;
  ofstream ofs("C:\\dev\\TCMM\\problems\\MM126\\out\\42.out");
  ostream& out = ofs;
#else
  istream& in = cin;
  ostream& out = cout;
#endif

  bool verbose = false;
  init(in);
  dump(N, C, H);

  State s = State::create(verbose);

  s.print_stat(cerr);

  dump(s.distance_cost);

  s.solve_hole_in_one();
  dump(s.raw_score);
  s.print_stat(cerr);

  //s.vis();

  dump(s.distance_cost);

  s.solve_greedy();

  //s.vis();

  dump(s.distance_cost);

  s.print_stat(cerr);

  s.output(out);

  return 0;
}