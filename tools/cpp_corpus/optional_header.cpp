#include <cstdio>
#include <optional>
static std::optional<int> half(int n) {
    if (n % 2 == 0) { return std::optional<int>(n / 2); }
    return std::optional<int>();
}
int main() {
    std::optional<int> a = half(10);
    std::optional<int> b = half(7);
    printf("%d %d %d %d\n", (int)a.has_value(), a.value(), (int)b.has_value(), b.value_or(-1));
    return 0;
}
