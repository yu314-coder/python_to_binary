#include <cstdio>
#include <vector>
static int live = 0;
struct E { int c; E(int v) : c(v) {} };
struct R { int v; R(int x) : v(x) { if (x < 0) throw E(x); ++live; } ~R() { --live; } };
int main() {
    int got = 0;
    { std::vector<R> v; v.push_back(R(1)); v.push_back(R(2));
      try { v.push_back(R(-1)); } catch (const E &e) { got = e.c; }
      printf("%d %d\n", (int)v.size(), got); }
    printf("%d\n", live);
    return 0;
}
