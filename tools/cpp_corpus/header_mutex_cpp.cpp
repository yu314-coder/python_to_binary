/* <mutex>: a lock_guard held while the shared number is changed. */
#include <mutex>
#include <thread>
#include <cstdio>

static std::mutex gate;
static int total = 0;

static void add(int by) {
    std::lock_guard<std::mutex> held(gate);
    total += by;
}

int main() {
    std::thread one(add, 3);
    std::thread two(add, 4);
    one.join();
    two.join();
    printf("%d\n", total);
    return 0;
}
