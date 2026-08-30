#include <cstdio>
#include <functional>
static std::function<int(int)> adder(int n) { return [n](int v) { return v + n; }; }
int main() {
    int seen = 0;
    auto bump = [&seen](int v) { seen += v; };
    bump(3); bump(4);
    auto plus5 = adder(5);
    int total = 0;
    auto twice = [&](int v) { total = plus5(v) + plus5(v); };
    twice(1);
    printf("%d %d %d\n", seen, plus5(10), total);
    return 0;
}
