/* <chrono>: two readings of a clock and the gap between them, cast down to
   each unit. Only what every implementation promises is printed - that time
   does not go backwards, and that a coarser unit is no larger. */
#include <chrono>
#include <cstdio>

int main() {
    auto began = std::chrono::steady_clock::now();
    auto ended = std::chrono::steady_clock::now();
    long long apart = std::chrono::duration_cast<std::chrono::nanoseconds>(
        ended - began).count();
    long long coarser = std::chrono::duration_cast<std::chrono::microseconds>(
        ended - began).count();
    printf("%d %d\n", apart >= 0, coarser <= apart);
    auto wall = std::chrono::system_clock::now();
    auto fine = std::chrono::high_resolution_clock::now();
    printf("%d\n", (wall - wall).count() == 0 && (fine - fine).count() == 0);
    return 0;
}
