#include <cstdio>
struct Cache {
    mutable int asked;
    int v;
    Cache(int x) : asked(0), v(x) {}
    int get() const { ++asked; return v; }
    int seen() const { return asked; }
};
static int sum(const Cache &c) { return c.get() + c.get(); }
int main() {
    Cache c(4);
    int s = sum(c);
    printf("%d %d\n", s, c.seen());
    return 0;
}
