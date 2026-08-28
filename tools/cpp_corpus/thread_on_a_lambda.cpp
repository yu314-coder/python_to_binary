#include <cstdio>
#include <thread>
#include <atomic>
static std::atomic<long> total;
static void plain() { for (int i = 0; i < 30000; i++) total.fetch_add(2); }
int main() {
    total.store(0);
    auto job = [] { for (int i = 0; i < 30000; i++) total.fetch_add(1); };
    std::thread a(job);
    std::thread b(job);
    a.join(); b.join();
    printf("%ld\n", total.load());
    return 0;
}
