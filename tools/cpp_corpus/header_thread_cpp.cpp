/* <thread>: two of them, each adding to the same counter and joined before
   it is read - so the answer does not depend on which ran first. */
#include <thread>
#include <atomic>
#include <cstdio>

static std::atomic<int> total(0);

static void add(int by) { total.fetch_add(by); }

int main() {
    std::thread one(add, 3);
    std::thread two(add, 4);
    one.join();
    two.join();
    printf("%d\n", total.load());
    return 0;
}
