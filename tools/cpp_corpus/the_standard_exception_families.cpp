// The `<stdexcept>` classes caught through their bases - `logic_error`, `runtime_error`, `exception` - each saying what it was given.
#include <cstdio>
#include <stdexcept>
int main() {
    int n = 0;
    try { throw std::invalid_argument("arg"); } catch (const std::logic_error &e) { n += 1; printf("%s\n", e.what()); }
    try { throw std::out_of_range("range"); } catch (const std::exception &e) { n += 10; printf("%s\n", e.what()); }
    try { throw std::overflow_error("over"); } catch (const std::runtime_error &e) { n += 100; printf("%s\n", e.what()); }
    try { throw std::length_error("len"); } catch (const std::invalid_argument &e) { n += 1000; } catch (const std::logic_error &e) { n += 10000; }
    printf("%d\n", n);
    return 0;
}
