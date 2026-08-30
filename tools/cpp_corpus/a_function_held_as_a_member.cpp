#include <cstdio>
#include <functional>
struct Bus {
    std::function<int(int)> on;
    void set(std::function<int(int)> f) { on = f; }
    int fire(int v) { return on(v); }
};
static int twice(int v) { return v * 2; }
int main() {
    Bus b;
    b.set([](int v) { return v + 10; });
    int one = b.fire(5);
    b.set(twice);
    printf("%d %d\n", one, b.fire(5));
    return 0;
}
