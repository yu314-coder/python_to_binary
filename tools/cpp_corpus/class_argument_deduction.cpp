#include <cstdio>
#include <mutex>
static std::mutex m;
static int total = 0;
static void bump() { std::lock_guard lock(m); total += 1; }
int main() { bump(); bump(); printf("%d\n", total); return 0; }
