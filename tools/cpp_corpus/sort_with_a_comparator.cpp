#include <cstdio>
#include <vector>
#include <algorithm>
struct P { int k; P(int v) : k(v) {} };
static bool before(const P &a, const P &b) { return a.k < b.k; }
int main() {
    std::vector<P> v;
    v.push_back(P(3)); v.push_back(P(1)); v.push_back(P(2));
    std::sort(v.begin(), v.end(), before);
    printf("%d %d %d\n", v[0].k, v[1].k, v[2].k);
    return 0;
}
