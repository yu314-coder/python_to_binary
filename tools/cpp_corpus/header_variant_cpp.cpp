/* <variant>: which of the two it holds, and getting that one out. */
#include <variant>
#include <cstdio>

int main() {
    std::variant<int, double> held = 3;
    printf("%d %d\n", (int)held.index(), std::get<int>(held));
    held = 1.5;
    printf("%d %.1f\n", (int)held.index(), std::get<double>(held));
    return 0;
}
