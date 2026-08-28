#include <cstdio>
int main() {
    auto twice = [](auto v) { return v + v; };
    printf("%d %.1f\n", twice(3), twice(1.5));
    return 0;
}
