// A class deriving from `std::exception` answers `what()` with its own text through a handler for the base.
#include <cstdio>
#include <exception>
struct MyErr : std::exception { const char *what() const noexcept override { return "mine"; } };
int main() {
    try { throw MyErr(); } catch (const std::exception &e) { printf("%s\n", e.what()); }
    return 0;
}
