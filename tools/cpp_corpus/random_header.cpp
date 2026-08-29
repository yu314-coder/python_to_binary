#include <cstdio>
#include <random>
int main() {
    std::mt19937 g(5489);
    unsigned long a = g(), b = g(), c = g();
    std::mt19937 h(7);
    std::uniform_int_distribution<int> d(1, 6);
    int rolls = 0;
    for (int i = 0; i < 100; i++) { int r = d(h); if (r < 1 || r > 6) rolls++; }
    printf("%lu %lu %lu %d\n", a, b, c, rolls);
    return 0;
}
