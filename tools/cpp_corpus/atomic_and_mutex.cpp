#include <cstdio>
#include <atomic>
#include <mutex>
int main() {
    std::atomic<bool> running(false);
    std::atomic<long> count(0);
    bool was = running.exchange(true);
    count.fetch_add(5);
    count.fetch_add(7);
    std::mutex m;
    int inside = 0;
    { std::lock_guard<std::mutex> hold(m); inside = 1; }
    int free_after = m.try_lock();
    m.unlock();
    printf("%d %d %ld %d %d\n", (int)was, (int)running.load(), count.load(), inside, free_after);
    return 0;
}
