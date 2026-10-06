// Four threads taking and giving back storage at once. The lists of blocks
// given back are shared, so they are reached under a lock - one waited for by
// reading, because waiting with the atomic add wrote the word on every turn and
// kept the thread handing the lock on from ever finishing its own add.
#include <cstdio>
#include <thread>
#include <vector>
static long totals[4];
static void work(int id) {
    long sum = 0;
    for (int round = 0; round < 200000; round++) {
        int *p = new int[1 + (round % 50)];
        p[0] = round;
        sum += p[0] % 7;
        delete[] p;
    }
    totals[id] = sum;
}
int main() {
    std::thread a(work, 0); std::thread b(work, 1); std::thread c(work, 2); std::thread d(work, 3);
    a.join(); b.join(); c.join(); d.join();
    printf("%ld %ld %ld %ld\n", totals[0], totals[1], totals[2], totals[3]);
    return 0;
}
