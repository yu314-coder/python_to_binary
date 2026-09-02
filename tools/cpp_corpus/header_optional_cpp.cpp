/* <optional>: empty, then holding something. */
#include <optional>
#include <cstdio>

int main() {
    std::optional<int> maybe;
    printf("%d\n", (int)maybe.has_value());
    maybe = 5;
    printf("%d %d %d\n", (int)maybe.has_value(), *maybe, maybe.value());
    return 0;
}
