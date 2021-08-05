#define NDEBUG
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include "bits/stdc++.h"
#include <unordered_map>
#include <unordered_set>
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
  double elapsedMs() { return (time() - t - paused) * 1000.0; }
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

namespace NVis {
  class Graphics {
  public:
    double screenW;
    double screenH;
    double infoH;
    ostringstream data;

    double sr;
    double sg;
    double sb;
    double sa;
    double sthick;
    double fr;
    double fg;
    double fb;
    double fa;

    Graphics() : screenW(1), screenH(1), sr(0), sg(0), sb(0), sa(1), sthick(1.0), fr(1), fg(1), fb(1), fa(1) {}

    void screen(int width, int height) {
      screenW = width;
      screenH = height;
      infoH = height / 20;
    }

    void clear() {
      data.str("");
      data.clear(stringstream::goodbit);
    }

    void stroke(double r, double g, double b, double thickness=3.0) {
      stroke(r, g, b, 1, thickness);
    }

    void stroke(double r, double g, double b, double a, double thickness) {
      sr = r;
      sg = g;
      sb = b;
      sa = a;
      sthick = thickness;
    }

    void noStroke() {
      stroke(0, 0, 0, 0);
    }

    void fill(double r, double g, double b) {
      fill(r, g, b, 1);
    }

    void fill(double r, double g, double b, double a) {
      fr = r;
      fg = g;
      fb = b;
      fa = a;
    }

    void noFill() {
      fill(0, 0, 0, 0);
    }

    void line(double x1, double y1, double x2, double y2) {
      data << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 << "\" y2=\"" << y2 << "\" " << stroke() << "/>\n";
    }

    void rect(double x, double y, double w, double h) {
      data << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << w << "\" height=\"" << h << "\" " << stroke() << " " + fill() << "/>\n";
    }

    void text(string str, double x, double y, double size = 16) {
      data << "<text text-anchor=\"middle\" x=\"" << x << "\" y=\"" << y << "\" font-size=\"" << size << "\" " << fill() << " >" << str << "</text>\n";
    }

    string dumps(string id = "", string style = "", int widthPx = -1, int heightPx = -1) const {
      ostringstream res;
      res << "<svg ";
      if (id != "") res << "id=\"" + id + "\" ";
      if (style != "") res << "style=\"" + style + "\" ";
      if (widthPx != -1) res << "width=\"" << widthPx << "\" ";
      if (heightPx != -1) res << "height=\"" << heightPx << "\" ";
      res << "viewBox=\"-1 -1 " << screenW + 2 << " " << screenH + infoH + 2 << "\" xmlns=\"http://www.w3.org/2000/svg\">\n" << data.str() << "</svg>";
      return res.str();
    }

  private:
    string stroke() const {
      return "stroke=\"" + rgb(sr, sg, sb) + "\" stroke-opacity=\"" + s(sa) + "\" stroke-width=\"" + s(sthick) + "\"";
    }

    string fill() const {
      return "fill=\"" + rgb(fr, fg, fb) + "\" fill-opacity=\"" + s(fa) + "\"";
    }

    string rgb(double r, double g, double b) const {
      return "rgb(" + s(lround(r * 255)) + "," + s(lround(g * 255)) + "," + s(lround(b * 255)) + ")";
    }

    string s(double x) const {
      return to_string(x);
    }
  } g;

  class Movie {
  public:
    vector<string> svgs;

    Movie() {}

    void clear() {
      svgs.clear();
    }

    void addFrame(Graphics& g) {
      svgs.push_back(g.dumps("f" + to_string(svgs.size()), "display:none;pointer-events:none;user-select:none;"));
    }

    string dumpHtml(int fps) {
      ostringstream res;
      res << "<html><body><div id=\"text\">loading...</div>" << endl;
      for (string& svg : svgs) {
        res << svg << endl;
      }

      res << "<script>\nlet numFrames = " << svgs.size() << ", fps = " << fps << ";";
      string script = 
R"(
	let text = document.getElementById("text");
	let frames = [];
	for (let i = 0; i < numFrames; i++) {
		let f = document.getElementById("f" + i);
		frames.push(f);
		f.style.display = "none";
	}
	let currentFrame = 0;
	let playing = true;
	setInterval(() => {
		if (!playing) return;
		text.innerText = (currentFrame + 1) + " / " + numFrames;
		frames[(currentFrame - 1 + numFrames) % numFrames].style.display = "none";
		frames[currentFrame].style.display = null;
		currentFrame = (currentFrame + 1) % numFrames;
		if (currentFrame == 0) playing = false;
	}, 1000 / fps);
	window.onmousedown = e => { if (e.button == 0) playing = true; };
;)";
      res << script;
      res << "</script>" << endl;
      res << "</body></html>" << endl;
      return res.str();
    }
  private:
  } mov;
}

constexpr int dy[] = { 0, -1, 0, 1 };
constexpr int dx[] = { 1, 0, -1, 0 };
constexpr int BSIZE = 10000;

int debug_count = 0;

namespace LocalSearchSolver {

  struct Deform {
    int rid, dir, inflate;
    double sdiff;
    int adiff;
    Deform() {}
    Deform(int rid, int dir, int inflate) : rid(rid), dir(dir), inflate(inflate) {}
    static Deform random(int N, int range_begin, int range_end, Xorshift& rnd) {
      Deform t;
      t.rid = rnd.next_int(N);
      t.dir = rnd.next_int(4);
      t.inflate = rnd.next_int(range_end - range_begin) + range_begin;
      t.sdiff = 0;
      t.adiff = 0;
      return t;
    }
    std::string str() const {
      return format("Deform [rid=%d, dir=%d, inflate=%d, sdiff=%.10f, adiff=%d]", rid, dir, inflate, sdiff, adiff);
    }
    friend std::ostream& operator<<(std::ostream& o, const Deform& obj) {
      o << obj.str();
      return o;
    }
  };

  struct ForceDeform {
    int rid, dir, inflate;
    double sdiff;
    int adiff;
    vector<Deform> seq;
    ForceDeform() {}
    ForceDeform(int rid, int dir, int inflate) : rid(rid), dir(dir), inflate(inflate) {}
    static ForceDeform random(int N, int range, Xorshift& rnd) {
      assert(range > 0);
      ForceDeform t;
      t.rid = rnd.next_int(N);
      t.dir = rnd.next_int(4);
      t.inflate = rnd.next_int(range) + 1;
      t.sdiff = 0.0;
      t.adiff = 0;
      return t;
    }
    std::string str() const {
      return format("ForceDeform [rid=%d, dir=%d, inflate=%d, diff=%.10f, adiff=%d, seq.size()=%lld]", rid, dir, inflate, sdiff, adiff, seq.size());
    }
    friend std::ostream& operator<<(std::ostream& o, const ForceDeform& obj) {
      o << obj.str();
      return o;
    }
  };

  struct Rect {
    int id;
    int tx, ty, tr; // center
    int x1, y1; // top-left
    int x2, y2; // bottom-right
    int area;
    double score;
    Rect() {}
    Rect(int id, int tx, int ty, int tr, int x1, int y1, int x2, int y2) : id(id), tx(tx), ty(ty), tr(tr), x1(x1), y1(y1), x2(x2), y2(y2) {
      area = (x2 - x1) * (y2 - y1);
      score = 1.0 - pow(1.0 - (double)min(tr, area) / max(tr, area), 2.0);
    }
    void reset(int x1, int y1, int x2, int y2) {
      this->x1 = x1; this->y1 = y1; this->x2 = x2; this->y2 = y2;
      area = (x2 - x1) * (y2 - y1);
      double z = 1.0 - (double)min(tr, area) / max(tr, area);
      score = 1.0 - z * z;
    }
    void reset() {
      this->x1 = tx; this->x2 = tx + 1;
      this->y1 = ty; this->y2 = ty + 1;
      area = 1;
      double z = 1.0 - (double)min(tr, area) / max(tr, area);
      score = 1.0 - z * z;
    }
    void avoid(const Rect& r) {
      int min_diff = INT_MAX;
      int mx1, my1, mx2, my2;
      auto [ix1, iy1, ix2, iy2] = get_intersect(r);
      if (ix1 < x2 && tx < ix1) { // x2 を ix1 に合わせる
        int diff = abs(tr - (ix1 - x1) * (y2 - y1));
        if (diff < min_diff) {
          min_diff = diff;
          mx1 = x1; mx2 = ix1; my1 = y1; my2 = y2;
        }
      }
      if (iy1 < y2 && ty < iy1) { // y2 を iy2 に合わせる
        int diff = abs(tr - (x2 - x1) * (iy1 - y1));
        if (diff < min_diff) {
          min_diff = diff;
          mx1 = x1; mx2 = x2; my1 = y1; my2 = iy1;
        }
      }
      if (x1 < ix2 && ix2 <= tx) { // x1 を ix2 に合わせる
        int diff = abs(tr - (x2 - ix2) * (y2 - y1));
        if (diff < min_diff) {
          min_diff = diff;
          mx1 = ix2; mx2 = x2; my1 = y1; my2 = y2;
        }
      }
      if (y1 < iy2 && iy2 <= ty) { // y1 を iy2 に合わせる
        int diff = abs(tr - (x2 - x1) * (y2 - iy2));
        if (diff < min_diff) {
          min_diff = diff;
          mx1 = x1; mx2 = x2; my1 = iy2; my2 = y2;
        }
      }
      reset(mx1, my1, mx2, my2);
    }
    tuple<int, int, int, int> get_intersect(const Rect& r) const {
      return { max(x1, r.x1), max(y1, r.y1), min(x2, r.x2), min(y2, r.y2) };
    }
    bool has_intersect(const Rect& r) const {
      return max(x1, r.x1) < min(x2, r.x2) && max(y1, r.y1) < min(y2, r.y2);
    }
    tuple<bool, int, int> intersect_info(const Rect& r) const {
      int ox = max(0, min(x2, r.x2) - max(x1, r.x1));
      int oy = max(0, min(y2, r.y2) - max(y1, r.y1));
      bool intersect = ox * oy > 0;
      return { intersect, ox, oy };
    }
    bool is_acceptable(Deform& t) const {
      bool ok;
      int narea;
      if (t.dir == 0) {
        int x = x2 + t.inflate;
        ok = (tx < x) && (x <= BSIZE);
        narea = (x - x1) * (y2 - y1);
      }
      else if (t.dir == 1) {
        int y = y1 - t.inflate;
        ok = (0 <= y) && (y <= ty);
        narea = (x2 - x1) * (y2 - y);
      }
      else if (t.dir == 2) {
        int x = x1 - t.inflate;
        ok = (0 <= x) && (x <= tx);
        narea = (x2 - x) * (y2 - y1);
      }
      else if (t.dir == 3) {
        int y = y2 + t.inflate;
        ok = (ty < y) && (y <= BSIZE);
        narea = (x2 - x1) * (y - y1);
      }
      else {
        assert(false);
      }
      double a = 1.0 - (double)min(tr, narea) / max(tr, narea), nscore = 1.0 - a * a;
      t.sdiff = nscore - score;
      t.adiff = narea - area;
      return ok;
    }
    bool is_acceptable(ForceDeform& t) const {
      Deform t2(t.rid, t.dir, t.inflate);
      bool ok = is_acceptable(t2);
      if (t2.sdiff < 0) return false;
      t.sdiff = t2.sdiff;
      t.adiff = t2.adiff;
      t.seq.push_back(t2);
      return ok;
    }
    void accept(const Deform& t) {
      if (t.dir == 0) x2 += t.inflate;
      else if (t.dir == 1) y1 -= t.inflate;
      else if (t.dir == 2) x1 -= t.inflate;
      else if (t.dir == 3) y2 += t.inflate;
      area += t.adiff;
      score += t.sdiff;
    }
    void accept(const ForceDeform& t) {
      if (t.dir == 0) x2 += t.inflate;
      else if (t.dir == 1) y1 -= t.inflate;
      else if (t.dir == 2) x1 -= t.inflate;
      else if (t.dir == 3) y2 += t.inflate;
      area += t.adiff;
      score += t.sdiff;
    }

    std::string str() const {
      return format("Rect [id=%d, tx=%d, ty=%d, tr=%d, x1=%d, y1=%d, x2=%d, y2=%d, area=%d, score=%.10f]", id, tx, ty, tr, x1, y1, x2, y2, area, score);
    }
    friend std::ostream& operator<<(std::ostream& o, const Rect& obj) {
      o << obj.str();
      return o;
    }
    tuple<double, double, double> get_color() const {
      double ratio = double((x2 - x1) * (y2 - y1)) / tr;
      if (ratio < 1.0) {
        return { 1.0, 0.0, ratio };
      }
      if (ratio < 2.0) {
        return { 2.0 - ratio, 0.0, 1.0 };
      }
      return { 0.0, 0.0, 1.0 };
    }
    void draw_rect(NVis::Graphics& g) const {
      auto [b_, g_, r_] = get_color();
      g.stroke(0, 0, 0);
      g.fill(r_, g_, b_);
      g.rect(x1, y1, x2 - x1, y2 - y1);
    }
    void draw_info(NVis::Graphics& g) const {
      g.stroke(0, 0, 0);
      g.fill(0, 0, 0);
      g.text(format("%d %.3f", id, score), (x1 + x2) * 0.5, (y1 + y2) * 0.5 + 50.0, 100.0);
    }
    void draw_point(NVis::Graphics& g) const {
      g.stroke(0, 0, 0);
      g.fill(0, 0, 0);
      double cx = (x1 + x2) * 0.5, cy = (y1 + y2) * 0.5;
      g.line(tx, ty, cx, cy);
    }
  };

  struct State {
    int N;
    vector<Rect> rects;
    double score;
    State() {}
    State(int N, const vector<Rect>& rects) : N(N), rects(rects) {
      score = 0.0;
      for (const auto& r : rects) {
        score += r.score;
      }
    }
    bool is_valid() const {
      for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
          if (rects[i].has_intersect(rects[j])) return false;
        }
      }
      return true;
    }
    bool is_acceptable(Deform& t) const {
      // single rectangle check
      auto ok = rects[t.rid].is_acceptable(t); // modify t
      if (!ok) return false;
      // intersection check
      Rect rect = rects[t.rid];
      rect.accept(t);
      for (int rid = 0; rid < N; rid++) {
        if (rid == t.rid) continue;
        if (rect.has_intersect(rects[rid])) return false;
      }
      return true;
    }
    bool is_acceptable(ForceDeform& t) const {
      // single rectangle check
      bool ok = rects[t.rid].is_acceptable(t);
      if (!ok) return false;
      // intersection check
      Rect rect = rects[t.rid];
      rect.accept(t);
      for (int rid = 0; rid < N; rid++) {
        if (rid == t.rid) continue;
        auto [overlap, ox, oy] = rects[rid].intersect_info(rect);
        if (overlap) {
          // rect の膨張によって overlap が発生する
          int o = (t.dir % 2 == 0) ? ox : oy;
          assert(o > 0);
          Deform t2(rid, (t.dir + 2) & 3, -o); // shrink
          bool ok2 = rects[rid].is_acceptable(t2);
          if (!ok2) return false;
          t.sdiff += t2.sdiff;
          t.seq.push_back(t2);
        }
      }
      return true;
    }
    void accept(const Deform& t) {
      rects[t.rid].accept(t);
      score += t.sdiff;
    }
    void accept(const ForceDeform& t) {
      for (const Deform& t2 : t.seq) {
        rects[t2.rid].accept(t2);
        score += t2.sdiff;
      }
    }

    void output(ostream& out) const {
      for (int i = 0; i < N; i++) {
        out << rects[i].x1 << ' ' << rects[i].y1 << ' ' << rects[i].x2 << ' ' << rects[i].y2 << '\n';
      }
    }
    int area_sum() const {
      int asum = 0;
      for (int i = 0; i < N; i++) asum += rects[i].area;
      return asum;
    }
    void vis(NVis::Graphics& g, NVis::Movie& mov, double time, const string& func) const {
      g.clear();
      for (const auto& r : rects) r.draw_rect(g);
      for (const auto& r : rects) r.draw_point(g);
      for (const auto& r : rects) r.draw_info(g);
      ll lscore = llround(score * 1e9 / N);
      string info = format("time:%3.2f / score:%lld / @%s", time, lscore, func.c_str());
      g.fill(0, 0, 0);
      g.text(info, g.screenW / 2, g.screenH + g.infoH / 2, g.infoH / 3);
      mov.addFrame(g);
    }
  };

  struct Solver {
    State best_state;
    State state;
    Solver(const State& state) : best_state(state), state(state) {}

    void annealing(double process_ms, double begin_temp, double end_temp) {
      auto get_temp = [](double stemp, double etemp, double loop, double num_loop) {
        return etemp + (stemp - etemp) * (num_loop - loop) / num_loop;
      };
      auto get_width = [](double tbegin, double tend, double t) {
        return 10.0 + (500.0 - 10.0) * (tend - t) / (tend - tbegin);
      };
      double tstart = timer.elapsedMs(), t = tstart, tend = tstart + process_ms;
      int width = (int)get_width(tstart, tend, tstart);
      int loop = 0;
      while (true) {
        // deform
        Deform trans = Deform::random(state.N, -width, width, rnd);

        auto ok = state.is_acceptable(trans);
        if (!ok) continue;

        double temp = get_temp(begin_temp, end_temp, t - tstart, tend - tstart);
        double prob = exp(trans.sdiff / temp);

        if (rnd.next_double() < prob) {
          state.accept(trans);
          if (best_state.score < state.score) {
            best_state = state;
          }
        }
        loop++;
        if (!(loop & 0xFFFFF)) {
          ll lls = (ll)round(state.score * 1e9 / state.N);
          dump(t, loop, best_state.score, state.score, state.N, lls, state.area_sum(), width);
        }
        if (!(loop & 0xFFF)) {
          timer.pause();
          state.vis(NVis::g, NVis::mov, t, __FUNCTION__);
          timer.restart();
        }
        if (!(loop & 0xFF)) {
          t = timer.elapsedMs();
          width = (int)get_width(tstart, tend, t);
          if (t > tend) break;
        }
      }
      dump(t, loop, best_state.score, state.score, state.area_sum());
    }

    void climbing(double process_ms) {
      double tend = timer.elapsedMs() + process_ms;
      int loop = 0;
      while (timer.elapsedMs() < tend) {
        if (loop % 20 == 0) {
          double t = timer.elapsedMs();
          timer.pause();
          best_state.vis(NVis::g, NVis::mov, t, __FUNCTION__);
          timer.restart();
        }
        for (int rid = 0; rid < best_state.N; rid++) {
          for (int d = 0; d < 4; d++) {
            ForceDeform trans(rid, d, 1);
            auto ok = best_state.is_acceptable(trans);
            if (!ok) continue;
            if (trans.sdiff > 0) {
              best_state.accept(trans);
            }
          }
        }
        loop++;
      }
    }
  };

} // LocalSearchSolver

namespace SplitSolver {

  struct Point {
    int id, x, y, r;
    Point(int id = -1, int x = 0, int y = 0, int r = 0) : id(id), x(x), y(y), r(r) {}
    std::string str() const {
      return format("Point [id=%d, x=%d, y=%d, r=%d]", id, x, y, r);
    }
    friend std::ostream& operator<<(std::ostream& o, const Point& obj) {
      o << obj.str();
      return o;
    }
  };

  struct Rect {
    int x1, y1, x2, y2;
    Rect(int x1 = 0, int y1 = 0, int x2 = 0, int y2 = 0) : x1(x1), y1(y1), x2(x2), y2(y2) {}
    int area() const { return (x2 - x1) * (y2 - y1); }
    std::string str() const {
      return format("Rect [x1=%d, y1=%d, x2=%d, y2=%d]", x1, y1, x2, y2);
    }
    friend std::ostream& operator<<(std::ostream& o, const Rect& obj) {
      o << obj.str();
      return o;
    }
  };

  struct Area {
    Rect rect;
    vector<Point> points;
    shared_ptr<Area> left, right;
    Area() {}
    Area(const Rect& rect) : rect(rect) {}
    Area(const Rect& rect, const vector<Point>& points) : rect(rect), points(points) {}
    Area(int x1, int y1, int x2, int y2)
      : rect(x1, y1, x2, y2), left(nullptr), right(nullptr) {}
    Area(int x1, int y1, int x2, int y2, const vector<Point>& points)
      : rect(x1, y1, x2, y2), points(points), left(nullptr), right(nullptr) {}
    int calc_r() const {
      int r = 0;
      for (const auto& p : points) r += p.r;
      return r;
    }
    int calc_s() const { return rect.area(); }
    double calc_p() const {
      int r = calc_r(), s = calc_s();
      double a = 1.0 - double(min(r, s)) / max(r, s);
      return 1.0 - a * a;
    }
    std::string str() const {
      return format(
        "Area [rect=%s, points.size()=%lld, r=%d, s=%d, p=%.10f]",
        rect.str().c_str(), points.size(), calc_r(), calc_s(), calc_p()
      );
    }
    friend std::ostream& operator<<(std::ostream& o, const Area& obj) {
      o << obj.str();
      return o;
    }
    void vsplit(int x) {
      assert(!left && !right);
      left = make_shared<Area>(rect.x1, rect.y1, x, rect.y2);
      right = make_shared<Area>(x, rect.y1, rect.x2, rect.y2);
      for (const Point& p : points) {
        (p.x < x ? left->points : right->points).push_back(p);
      }
    }
    void hsplit(int y) {
      assert(!left && !right);
      left = make_shared<Area>(rect.x1, rect.y1, rect.x2, y);
      right = make_shared<Area>(rect.x1, y, rect.x2, rect.y2);
      for (const Point& p : points) {
        (p.y < y ? left->points : right->points).push_back(p);
      }
    }
    map<double, int> enum_vsplit_candidates() const {
      int lr = 0, rr = 0;
      auto ps(points);
      sort(ps.begin(), ps.end(), [](const Point& a, const Point& b) { return a.x < b.x; });
      for (const auto& p : ps) rr += p.r;
      int np = ps.size();
      int ylen = rect.y2 - rect.y1;
      map<double, int> res;
      for (int i = 0; i < np - 1; i++) {
        const Point& p1 = ps[i];
        const Point& p2 = ps[i + 1];
        // 左に p1 を移動
        lr += ps[i].r; rr -= ps[i].r;
        if (p1.x == p2.x) continue;
        // [p1.x + 1, p2.x] の間を走査して最大値を見つける
        // TODO: O(1) 解法ないか？
        int xbest = -1;
        double sbest = -1.0;
        for (int x = p1.x + 1; x <= p2.x; x++) {
          int ls = ylen * (x - rect.x1), rs = ylen * (rect.x2 - x);
          double la = 1.0 - double(min(lr, ls)) / max(lr, ls), lp = 1.0 - la * la;
          double ra = 1.0 - double(min(rr, rs)) / max(rr, rs), rp = 1.0 - ra * ra;
          double s = (lp + rp) * 0.5;
          if (sbest < s) {
            sbest = s; xbest = x;
          }
        }
        res[sbest] = xbest;
      }
      return res;
    }
    map<double, int> enum_hsplit_candidates() const {
      int tr = 0, br = 0;
      auto ps(points);
      sort(ps.begin(), ps.end(), [](const Point& a, const Point& b) { return a.y < b.y; });
      for (const auto& p : ps) br += p.r;
      int np = ps.size();
      int xlen = rect.x2 - rect.x1;
      map<double, int> res;
      for (int i = 0; i < np - 1; i++) {
        const Point& p1 = ps[i];
        const Point& p2 = ps[i + 1];
        // 上に p1 を移動
        tr += ps[i].r; br -= ps[i].r;
        if (p1.y == p2.y) continue;
        // [p1.y + 1, p2.y] の間を走査して最小値を見つける
        // TODO: O(1) 解法ないか？
        int ybest = -1;
        double sbest = -1.0;
        for (int y = p1.y + 1; y <= p2.y; y++) {
          int ts = xlen * (y - rect.y1), bs = xlen * (rect.y2 - y);
          double ta = 1.0 - double(min(tr, ts)) / max(tr, ts), tp = 1.0 - ta * ta;
          double ba = 1.0 - double(min(br, bs)) / max(br, bs), bp = 1.0 - ba * ba;
          double s = (tp + bp) * 0.5;
          if (sbest < s) {
            sbest = s; ybest = y;
          }
        }
        res[sbest] = ybest;
      }
      return res;
    }
    void best_split() {
      assert(points.size() > 1);
      auto hsp = enum_hsplit_candidates();
      auto vsp = enum_vsplit_candidates();
      if (hsp.empty()) {
        vsplit(vsp.rbegin()->second);
        return;
      }
      if (vsp.empty()) {
        hsplit(hsp.rbegin()->second);
        return;
      }
      if (hsp.rbegin()->first < vsp.rbegin()->first) {
        vsplit(vsp.rbegin()->second);
        return;
      }
      else {
        hsplit(hsp.rbegin()->second);
        return;
      }
      assert(false);
    }
    void random_split(Xorshift& rnd, int width, double pruning_thresh) {
      // vsplit から width 個、hsplit から width 個選ぶ
      // pruning thresh 以下のものは(1 つを除いて)弾く
      // 候補からランダムに選択
      auto vsp = enum_vsplit_candidates();
      auto hsp = enum_hsplit_candidates();
      vector<tuple<double, int, int>> cands; // score, direction, pos
      int w = 0;
      for (auto it = vsp.rbegin(); it != vsp.rend(); ++it) {
        cands.emplace_back(it->first, 0, it->second);
        w++;
        if (w >= width) break;
      }
      w = 0;
      for (auto it = hsp.rbegin(); it != hsp.rend(); ++it) {
        cands.emplace_back(it->first, 1, it->second);
        w++;
        if (w >= width) break;
      }
      sort(cands.rbegin(), cands.rend());
      while (cands.size() > 1 && get<0>(cands.back()) < pruning_thresh) cands.pop_back();
      assert(cands.size());
      auto selected = cands[rnd.next_int(cands.size())];
      if (get<1>(selected) == 0) {
        // vsp
        vsplit(get<2>(selected));
      }
      else {
        // hsp
        hsplit(get<2>(selected));
      }
    }
    void adjust_bbox() {
      assert(points.size() == 1);
      // shrink して最もフィットする長方形を求める
      // TODO: O(1) 解法ないか？
      const auto& p = points.front();
      if (p.r > calc_s()) return;
      int xdim = rect.x2 - rect.x1;
      int ydim = rect.y2 - rect.y1;
      int dmin = INT_MAX, xbest = -1, ybest = -1;
      for (int x = 1; x <= xdim; x++) {
        int y = min(ydim, p.r / x);
        int d = abs(p.r - x * y);
        if (d < dmin) {
          dmin = d; xbest = x; ybest = y;
        }
      }
      for (int y = 1; y <= ydim; y++) {
        int x = min(xdim, p.r / y);
        int d = abs(p.r - x * y);
        if (d < dmin) {
          dmin = d; xbest = x; ybest = y;
        }
      }
      // 左上に p.x, p.y が来るようにして、後で調整
      int x1 = p.x, y1 = p.y, x2 = x1 + xbest, y2 = y1 + ybest;
      if (x2 > rect.x2) {
        int slide = x2 - rect.x2;
        x1 -= slide; x2 -= slide;
      }
      if (y2 > rect.y2) {
        int slide = y2 - rect.y2;
        y1 -= slide; y2 -= slide;
      }
      rect = Rect(x1, y1, x2, y2);
      assert(rect.x1 <= p.x && p.x < rect.x2&& rect.y1 <= p.y && p.y < rect.y2);
    }
  };
  using AreaPtr = shared_ptr<Area>;

  struct State {
    int N;
    AreaPtr root;
    vector<AreaPtr> leaves;
    State() {}
    State(int N, const vector<Point>& points) : N(N) {
      root = make_shared<Area>(0, 0, BSIZE, BSIZE, points);
    }
    void recursive_random_split(Xorshift& rnd, const AreaPtr& node, int width, double pruning_thresh) {
      if (node->points.size() == 1) {
        node->adjust_bbox();
        leaves.push_back(node);
        return;
      }
      node->random_split(rnd, width, pruning_thresh);
      if (node->left) {
        recursive_random_split(rnd, node->left, width, pruning_thresh);
      }
      if (node->right) {
        recursive_random_split(rnd, node->right, width, pruning_thresh);
      }
    }
    double calc_score() const {
      double score = 0.0;
      for (const auto& l : leaves) score += l->calc_p();
      return score;
    }
    vector<Rect> get_rects() const {
      vector<Rect> res;
      for (const auto& l : leaves) {
        res.push_back(l->rect);
      }
      return res;
    }
    LocalSearchSolver::State cvt() const {
      vector<LocalSearchSolver::Rect> arects;
      vector<pair<Point, Rect>> rects;
      for (const auto& p : leaves) {
        rects.emplace_back(p->points.front(), p->rect);
      }
      sort(rects.begin(), rects.end(), [](const auto& a, const auto& b) {
        return a.first.id < b.first.id;
      });
      for (const auto& [point, rect] : rects) {
        arects.emplace_back(point.id, point.x, point.y, point.r, rect.x1, rect.y1, rect.x2, rect.y2);
      }
      return LocalSearchSolver::State(N, arects);
    }
    tuple<double, double, double> get_color(const Point& p, const Rect& r) const {
      double ratio = double(r.area()) / p.r;
      if (ratio < 1.0) {
        return { 1.0, 0.0, ratio };
      }
      if (ratio < 2.0) {
        return { 2.0 - ratio, 0.0, 1.0 };
      }
      return { 0.0, 0.0, 1.0 };
    }
    double calc_rect_score(const Point& p, const Rect& r) const {
      int area = r.area();
      double z = 1.0 - (double)min(p.r, area) / max(p.r, area);
      return 1.0 - z * z;
    }
    void draw_rect(NVis::Graphics& g, const Point& p, const Rect& r) const {
      auto [b_, g_, r_] = get_color(p, r);
      g.stroke(0, 0, 0);
      g.fill(r_, g_, b_);
      g.rect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
    }
    void draw_point(NVis::Graphics& g, const Point& p, const Rect& r) const {
      g.stroke(0, 0, 0);
      g.fill(0, 0, 0);
      double cx = (r.x1 + r.x2) * 0.5, cy = (r.y1 + r.y2) * 0.5;
      g.line(p.x, p.y, cx, cy);
    }
    void draw_info(NVis::Graphics& g, const Point& p, const Rect& r) const {
      g.stroke(0, 0, 0);
      g.fill(0, 0, 0);
      double px = (r.x1 + r.x2) * 0.5, py = (r.y1 + r.y2) * 0.5 + 50.0;
      g.text(format("%d %.3f", p.id, calc_rect_score(p, r)), px, py, 100.0);
    }
    void vis(NVis::Graphics& g, NVis::Movie& mov, double time, const string& func) const {
      vector<pair<Point, Rect>> rects;
      for (const auto& p : leaves) {
        rects.emplace_back(p->points.front(), p->rect);
      }
      g.clear();
      for (const auto& r : rects) draw_rect(g, r.first, r.second);
      for (const auto& r : rects) draw_point(g, r.first, r.second);
      for (const auto& r : rects) draw_info(g, r.first, r.second);
      ll lscore = llround(calc_score() * 1e9 / N);
      string info = format("time:%3.2f / score:%lld / @%s", time, lscore, func.c_str());
      g.fill(0, 0, 0);
      g.text(info, g.screenW / 2, g.screenH + g.infoH / 2, g.infoH / 3);
      mov.addFrame(g);
    }
  };
  using StatePtr = shared_ptr<State>;

} // SplitSolver

#ifdef _MSC_VER
namespace ManualSolver {

  struct Rect {
    int id, tx, ty, tr, x1, y1, x2, y2;
    int area;
    double ratio;
    double score;
    Rect() {}
    Rect(int id, int tx, int ty, int tr, int x1, int y1, int x2, int y2)
      : id(id), tx(tx), ty(ty), tr(tr), x1(x1), y1(y1), x2(x2), y2(y2) {
      update();
    }
    Rect(int id, const LocalSearchSolver::Rect& r)
      : id(id), tx(r.tx), ty(r.ty), tr(r.tr), x1(r.x1), y1(r.y1), x2(r.x2), y2(r.y2) {
      update();
    }
    std::string str() const {
      return format(
        "Rect [id=%d, tx=%d, ty=%d, tr=%d, x1=%d, y1=%d, x2=%d, y2=%d, area=%d, ratio=%.10f, score=%.10f]",
        id, tx, ty, tr, x1, y1, x2, y2, area, ratio, score
      );
    }
    friend std::ostream& operator<<(std::ostream& o, const Rect& obj) {
      o << obj.str();
      return o;
    }
    bool has_intersect(const Rect& r) const {
      return max(x1, r.x1) < min(x2, r.x2) && max(y1, r.y1) < min(y2, r.y2);
    }
    bool is_inside(int x, int y) const {
      return x1 <= x && x <= x2 && y1 <= y && y <= y2;
    }
    void update() {
      area = (x2 - x1) * (y2 - y1);
      ratio = double(area) / tr; // ratio > 1: 過大
      double z = 1.0 - (double)min(tr, area) / max(tr, area);
      score = 1.0 - z * z;
    }
    cv::Scalar get_color() const {
      if (ratio < 1.0) {
        int red = (int)round(ratio * 255.0);
        return cv::Scalar(255, 0, red);
      }
      if (ratio < 2.0) {
        int blue = 255 - (int)round((ratio - 1.0) * 255.0);
        return cv::Scalar(blue, 0, 255);
      }
      return cv::Scalar(0, 0, 255);
    }
    cv::Mat_<cv::Vec3b> get_button_img(int size) const {
      auto color = get_color();
      cv::Mat_<cv::Vec3b> img(size, size, cv::Vec3b(150, 150, 150));
      cv::rectangle(img, cv::Rect(1, 1, size - 2, size - 2), color, cv::FILLED);
      cv::putText(
        img,
        to_string(id),
        cv::Point(size / 8, size / 5),
        cv::FONT_HERSHEY_SIMPLEX,
        0.3,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA
      );
      cv::putText(
        img,
        to_string(ratio),
        cv::Point(size / 8, size * 2 / 5),
        cv::FONT_HERSHEY_SIMPLEX,
        0.3,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA
      );
      cv::putText(
        img,
        to_string(score),
        cv::Point(size / 8, size * 3 / 5),
        cv::FONT_HERSHEY_SIMPLEX,
        0.3,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA
      );
      cv::putText(
        img,
        to_string(tr),
        cv::Point(size / 8, size * 4 / 5),
        cv::FONT_HERSHEY_SIMPLEX,
        0.3,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA
      );
      return img;
    }
  };

  struct State {
    static constexpr int BUTTON_SIZE = 60;
    static constexpr int PALETTE_WIDTH = 10;
    static constexpr int BOARD_SIZE = 1251;
    static constexpr int STAT_HEIGHT = 256;
    static constexpr int STAT_WIDTH = 512;

    static constexpr int KEY_UP = 2490368;
    static constexpr int KEY_DOWN = 2621440;
    static constexpr int KEY_RIGHT = 2555904;
    static constexpr int KEY_LEFT = 2424832;

    int N;
    vector<Rect> rects;

    int selected;
    bool inflate_mode;
    int shift_distance;
    cv::Mat_<cv::Vec3b> palette;
    cv::Mat_<cv::Vec3b> board;
    cv::Mat_<cv::Vec3b> stat;

    State() {}
    State(const vector<Rect>& rects) : N(rects.size()), rects(rects), selected(-1), shift_distance(1) {}
    State(const vector<tuple<int, int, int, int, int, int, int>>& rects) : N(rects.size()) {

    }

    double calc_score() const {
      double score = 0.0;
      for (const auto& r : rects) score += r.score;
      return score;
    }
    cv::Mat_<cv::Vec3b> create_stat_img() const {
      cv::Mat_<cv::Vec3b> img(STAT_WIDTH, STAT_HEIGHT, cv::Vec3b(50, 50, 50));
      cv::putText(img, format("score: %.6f", calc_score()), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      return img;
    }

    cv::Mat_<cv::Vec3b> create_palette_img() const {
      int size = BUTTON_SIZE;
      int N = rects.size();
      int W = PALETTE_WIDTH, H = (N + W - 1) / W;
      cv::Mat_<cv::Vec3b> img(H * size, W * size, cv::Vec3b(255, 255, 255));
      for (const auto& rect : rects) {
        int id = rect.id, r = id / W, c = id % W;
        cv::Rect roi(c * size, r * size, size, size);
        auto button_img = rect.get_button_img(size);
        if (rect.id == selected) {
          cv::rectangle(button_img, cv::Rect(0, 0, size, size), cv::Scalar(0, 255, inflate_mode ? 0 : 255), 3);
        }
        button_img.copyTo(img(roi));
      }
      return img;
    }

    cv::Mat_<cv::Vec3b> create_board_img() const {
      cv::Mat_<cv::Vec3b> img(BOARD_SIZE, BOARD_SIZE, cv::Vec3b(100, 100, 100));
      // rect
      for (const auto& r : rects) {
        int x1 = r.x1 / 8, y1 = r.y1 / 8, x2 = r.x2 / 8, y2 = r.y2 / 8;
        cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
        cv::rectangle(img, roi, r.get_color(), cv::FILLED);
        cv::rectangle(img, roi, cv::Scalar(50, 50, 50), 1);
      }
      // mark
      if (selected != -1) {
        const auto& r = rects[selected];
        int x1 = r.x1 / 8, y1 = r.y1 / 8, x2 = r.x2 / 8, y2 = r.y2 / 8;
        cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
        cv::rectangle(img, roi, cv::Scalar(0, 255, inflate_mode ? 0 : 255), 2);
      }
      // point
      for (const auto& r : rects) {
        int tx = r.tx / 8, ty = r.ty / 8;
        cv::circle(img, cv::Point(tx, ty), 2, cv::Scalar(0, 0, 0), cv::FILLED);
      }
      // line
      for (const auto& r : rects) {
        int x = (r.x1 + r.x2) / 16, y = (r.y1 + r.y2) / 16, tx = r.tx / 8, ty = r.ty / 8;
        cv::line(img, cv::Point(x, y), cv::Point(tx, ty), cv::Scalar(50, 50, 50), 1, 8);
      }
      // number
      for (const auto& r : rects) {
        int x = (r.x1 + r.x2) / 16, y = (r.y1 + r.y2) / 16;
        string sid = to_string(r.id);
        int slide = 2 + (sid.size() * 3);
        cv::putText(img, sid, cv::Point(x - slide, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
      }
      return img;
    }

    static void palette_callback(int e, int x, int y, int f, void* param) {
      State* state = static_cast<State*>(param);
      auto palette_cpy = state->palette.clone();
      int r = y / BUTTON_SIZE, c = x / BUTTON_SIZE;
      int id = r * PALETTE_WIDTH + c;
      if (0 <= id && id < state->N) {
        if (e == 4 || e == 5) {
          // clicked
          state->selected = id;
          state->inflate_mode = (e == 4);
          state->palette = state->create_palette_img();
          state->board = state->create_board_img();
          state->stat = state->create_stat_img();
          palette_cpy = state->palette.clone();
        }
        cv::Rect roi(c * BUTTON_SIZE, r * BUTTON_SIZE, BUTTON_SIZE, BUTTON_SIZE);
        cv::rectangle(palette_cpy, roi, cv::Scalar(0, 0, 0), 1);
      }
      cv::imshow("palette", palette_cpy);
      cv::imshow("board", state->board);
      cv::imshow("stat", state->stat);
      cerr << id << endl;
    }

    static void board_callback(int e, int x, int y, int f, void* param) {
      if (e != 4 && e != 5) return;
      State* state = static_cast<State*>(param);
      int cx = x * 8, cy = y * 8;
      int id = -1;
      for (const auto& r : state->rects) {
        if (r.is_inside(cx, cy)) {
          id = r.id;
        }
      }
      if (id == -1) return;
      state->selected = id;
      state->inflate_mode = (e == 4);
      state->palette = state->create_palette_img();
      state->board = state->create_board_img();
      cv::imshow("palette", state->palette);
      cv::imshow("board", state->board);
    }

    bool is_arrow_key(int code) const {
      return code == KEY_UP || code == KEY_DOWN || code == KEY_LEFT || code == KEY_RIGHT;
    }
    void up_arrow_key_pressed() {
      auto& r = rects[selected];
      if (inflate_mode) {
        r.y1 -= shift_distance;
        r.y1 = max(0, r.y1);
        // aviod collision
        for (int i = 0; i < N; i++) {
          if (i == selected) continue;
          if (r.has_intersect(rects[i])) {
            r.y1 = rects[i].y2;
          }
        }
      }
      else {
        r.y2 -= shift_distance;
        r.y2 = max(r.ty + 1, r.y2);
      }
    }
    void down_arrow_key_pressed() {
      auto& r = rects[selected];
      if (inflate_mode) {
        r.y2 += shift_distance;
        r.y2 = min(BSIZE, r.y2);
        // aviod collision
        for (int i = 0; i < N; i++) {
          if (i == selected) continue;
          if (r.has_intersect(rects[i])) {
            r.y2 = rects[i].y1;
          }
        }
      }
      else {
        r.y1 += shift_distance;
        r.y1 = min(r.ty, r.y1);
      }
    }
    void left_arrow_key_pressed() {
      auto& r = rects[selected];
      if (inflate_mode) {
        r.x1 -= shift_distance;
        r.x1 = max(0, r.x1);
        // aviod collision
        for (int i = 0; i < N; i++) {
          if (i == selected) continue;
          if (r.has_intersect(rects[i])) {
            r.x1 = rects[i].x2;
          }
        }
      }
      else {
        r.x2 -= shift_distance;
        r.x2 = max(r.tx + 1, r.x2);
      }
    }
    void right_arrow_key_pressed() {
      auto& r = rects[selected];
      if (inflate_mode) {
        r.x2 += shift_distance;
        r.x2 = min(BSIZE, r.x2);
        // aviod collision
        for (int i = 0; i < N; i++) {
          if (i == selected) continue;
          if (r.has_intersect(rects[i])) {
            r.x2 = rects[i].x1;
          }
        }
      }
      else {
        r.x1 += shift_distance;
        r.x1 = min(r.tx, r.x1);
      }
    }
    void arrow_key_pressed(int code) {
      if (selected == -1) return;
      dump(rects[selected]);
      switch (code) {
      case KEY_UP: up_arrow_key_pressed(); break;
      case KEY_DOWN: down_arrow_key_pressed(); break;
      case KEY_LEFT: left_arrow_key_pressed(); break;
      case KEY_RIGHT: right_arrow_key_pressed(); break;
      default: assert(false);
      }
      rects[selected].update();
      dump(rects[selected]);
      palette = create_palette_img();
      board = create_board_img();
      stat = create_stat_img();
      cv::imshow("palette", palette);
      cv::imshow("board", board);
      cv::imshow("stat", stat);
    }
    bool is_num_key(int code) const {
      return 48 <= code && code < 58;
    }
    void num_key_pressed(int code) {
      shift_distance = (1 << (code - 48));
      cerr << "shift_distance changed: " << shift_distance << endl;
    }

    void play() {
      palette = create_palette_img();
      board = create_board_img();
      stat = create_stat_img();
      cv::namedWindow("palette", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("board", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("stat", cv::WINDOW_AUTOSIZE);
      cv::setMouseCallback("palette", palette_callback, this);
      cv::setMouseCallback("board", board_callback, this);
      cv::imshow("palette", palette);
      cv::imshow("board", board);
      cv::imshow("stat", stat);

      while (true) {
        int c = cv::waitKeyEx(15);
        if (c == 27) break;
        else if (is_num_key(c)) {
          num_key_pressed(c);
        }
        else if (is_arrow_key(c)) {
          arrow_key_pressed(c);
        }
      }

    }
  };

} // ManualSolver
#endif



int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

#ifdef _MSC_VER
  ifstream ifs("tools\\in\\0019.txt");
  istream& in = ifs;
  ofstream ofs("tools\\out\\0019.txt");
  ostream& out = ofs;
#else
  istream& in = cin;
  ostream& out = cout;
#endif

  NVis::g.screen(10000, 10000);

  bool manual = false;
  double time_multiplier = 1.0;
  double time_split = 1500 * time_multiplier;
  double time_climbing = 2500 * time_multiplier;
  double time_annealing = 4500 * time_multiplier;

  /* input */
  int N;
  vector<tuple<int, int, int>> input;
  in >> N;
  for (int i = 0; i < N; i++) {
    int x, y, r;
    in >> x >> y >> r;
    input.emplace_back(x, y, r);
  }

#ifdef _MSC_VER
  /* manual */
  if (manual) {
    vector<LocalSearchSolver::Rect> rects;
    for (int id = 0; id < N; id++) {
      const auto& [tx, ty, tr] = input[id];
      rects.push_back(LocalSearchSolver::Rect(id, tx, ty, tr, tx, ty, tx + 1, ty + 1));
    }
    LocalSearchSolver::State s(N, rects);
    vector<ManualSolver::Rect> mrects;
    for (int i = 0; i < N; i++) {
      const auto& r = s.rects[i];
      mrects.emplace_back(i, r);
    }
    ManualSolver::State state(mrects);
    state.play();
  }
#endif

  /* split */
  vector<SplitSolver::Point> points;
  for (int id = 0; id < N; id++) {
    auto [x, y, r] = input[id];
    points.emplace_back(id, x, y, r);
  }

  vector<pair<double, SplitSolver::StatePtr>> svec;
  {
    double best = -1.0;
    int loop = 0;
    double t;
    while ((t = timer.elapsedMs()) < time_split) {
      SplitSolver::StatePtr stmp = make_shared<SplitSolver::State>(N, points);
      stmp->recursive_random_split(rnd, stmp->root, 10, 0.95 + rnd.next_double() * 0.04);
      timer.pause();
      stmp->vis(NVis::g, NVis::mov, t, __FUNCTION__);
      timer.restart();
      double score = stmp->calc_score();
      svec.emplace_back(score, stmp);
      if (best < score) {
        best = score;
        dump(timer.elapsedMs(), best);
      }
      loop++;
    }
    dump(best, loop);
  }
  dump(svec.size());
  sort(svec.begin(), svec.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

  /* climbing */
  double best_score = -1.0;
  LocalSearchSolver::State best_state;
  while (!svec.empty() && timer.elapsedMs() < time_climbing) {
    const auto [score, sstate] = svec.back(); svec.pop_back();
    LocalSearchSolver::State astate = sstate->cvt();
    LocalSearchSolver::Solver solver(astate);
    solver.climbing(100);
    if (best_score < solver.best_state.score) {
      best_score = solver.best_state.score;
      best_state = solver.best_state;
      ll lls = (ll)round(best_state.score * 1e9 / best_state.N);
      dump(score, best_state.score, N, lls);
    }
  }
  dump(best_state.score);

  /* annealing */
  LocalSearchSolver::Solver annealer(best_state);
  annealer.annealing(time_annealing - timer.elapsedMs(), 0.03, 0.0);
  dump(annealer.best_state.score, timer.elapsedMs());
  annealer.climbing(400);
  dump(annealer.best_state.score, timer.elapsedMs());
  dump((ll)round(annealer.best_state.score * 1e9 / N));
  annealer.best_state.output(out);

  {
    ofstream ofs("movie.html");
    ofs << NVis::mov.dumpHtml(30) << endl;
    ofs.close();
  }

  return 0;
}
