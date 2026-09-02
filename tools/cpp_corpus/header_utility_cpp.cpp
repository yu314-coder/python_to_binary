/* <utility>: a pair, made two ways, and the swap. */
#include <utility>
#include <cstdio>

int main() {
    std::pair<int, double> made = std::make_pair(3, 1.5);
    printf("%d %.1f\n", made.first, made.second);
    std::pair<int, int> written(7, 8);
    printf("%d %d\n", written.first, written.second);
    int a = 1;
    int b = 2;
    std::swap(a, b);
    printf("%d %d\n", a, b);
    return 0;
}
