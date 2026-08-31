#include <cstdio>
#include <thread>
#include <mutex>
static int total = 0;
static std::mutex guard;
static void add(int n) { for (int i = 0; i < n; ++i) { guard.lock(); ++total; guard.unlock(); } }
int main() {
    std::thread a(add, 1000);
    std::thread b(add, 1000);
    std::thread c(add, 1000);
    a.join(); b.join(); c.join();
    printf("%d\n", total);
    return 0;
}
