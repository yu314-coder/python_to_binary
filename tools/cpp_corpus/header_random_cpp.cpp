/* <random>: a seeded engine and a distribution. The numbers themselves are
   an implementation's own, so what is printed is what every implementation
   promises - that the engine moves, and that the distribution stays inside
   the range it was given. */
#include <random>
#include <cstdio>

int main() {
    std::mt19937 source(1234);
    unsigned first = source();
    unsigned second = source();
    printf("%d %d\n", first != second, first != 0);
    std::uniform_int_distribution<int> spread(1, 6);
    int inside = 1;
    for (int i = 0; i < 20; i++) {
        int roll = spread(source);
        if (roll < 1 || roll > 6) { inside = 0; }
    }
    printf("%d\n", inside);
    return 0;
}
