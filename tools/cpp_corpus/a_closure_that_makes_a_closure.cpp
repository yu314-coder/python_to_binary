#include <cstdio>
#include <functional>
int main() {
    int base = 100;
    auto outer = [base](int a) {
        return [base, a](int b) { return base + a + b; };
    };
    auto add5 = outer(5);
    int total = 0;
    auto each = [&total, &add5](int v) { total += add5(v); };
    each(1); each(2);
    printf("%d %d\n", add5(0), total);
    return 0;
}
