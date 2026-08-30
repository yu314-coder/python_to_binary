#include <cstdio>
#include <memory>
#include <utility>
struct N { int v; N(int x) : v(x) {} ~N() { } };
static std::unique_ptr<N> make(int x) { return std::unique_ptr<N>(new N(x)); }
int main() {
    std::unique_ptr<N> a = make(4);
    std::unique_ptr<N> b = std::move(a);
    int gone = (a.get() == 0) ? 1 : 0;
    printf("%d %d %d\n", b->v, gone, make(9)->v);
    return 0;
}
