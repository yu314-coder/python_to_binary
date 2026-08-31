#include <cstdio>
#include <vector>
static int live = 0;
struct C { int v; C() : v(1) { ++live; } C(const C &o) : v(o.v) { ++live; } ~C() { --live; } };
int main() {
    { std::vector<C> v; v.push_back(C()); v.push_back(C()); printf("%d %d\n", (int)v.size(), v[1].v); }
    printf("%d\n", live);
    return 0;
}
