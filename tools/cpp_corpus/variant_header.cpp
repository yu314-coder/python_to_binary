#include <cstdio>
#include <variant>
int main() {
    std::variant<int, double> a = 5;
    std::variant<int, double> b = 2.5;
    printf("%d %d %.1f %d %d\n", a.index(), std::get<int>(a), std::get<double>(b),
           (int)std::holds_alternative<int>(a), (int)std::holds_alternative<int>(b));
    return 0;
}
