#include <cstdio>
#include <chrono>
int main() {
    auto a = std::chrono::steady_clock::now();
    long long total = 0;
    for (int i = 0; i < 3000000; i++) total += i;
    auto b = std::chrono::steady_clock::now();
    long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    long long us = std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    printf("%d %d %d %d\n", total > 0 ? 1 : 0, ns > 0 ? 1 : 0,
           us == ns / 1000 ? 1 : 0, ms == ns / 1000000 ? 1 : 0);
    return 0;
}
