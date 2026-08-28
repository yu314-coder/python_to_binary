#include <cstdio>
#include <thread>
#include <atomic>
struct Host {
    std::atomic<long> seen;
    Host() { seen.store(0); }
    void run() { for (int i = 0; i < 50000; i++) seen.fetch_add(1); }
};
int main() {
    Host h;
    std::thread a(&Host::run, &h);
    std::thread b(&Host::run, &h);
    a.join();
    b.join();
    printf("%ld %d\n", h.seen.load(), (int)a.joinable());
    return 0;
}
