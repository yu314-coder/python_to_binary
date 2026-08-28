#include <cstdio>
#include <memory>
#include <utility>
struct Res { int id; Res(int n) : id(n) {} };
int main() {
    std::unique_ptr<Res> a(new Res(7));
    std::unique_ptr<Res> b = std::move(a);
    int have = (a == nullptr) ? 0 : 1;
    printf("%d %d\n", b->id, have);
    return 0;
}
