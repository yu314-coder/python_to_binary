// Storage given back is used again. The allocator handed out a fixed arena
// and its free() kept nothing, so a program that builds and drops a
// container in a loop ran out after a few hundred megabytes of turnover.
#include <cstdio>
#include <vector>
int main() {
    long total = 0;
    for (int round = 0; round < 20000; round++) {
        std::vector<long> held;
        for (int k = 0; k < 400; k++) held.push_back(round + k);
        total += held[399] - held[0];
        int *raw = new int[1000];
        raw[999] = round % 3;
        total += raw[999];
        delete[] raw;
    }
    printf("%ld\n", total);
    return 0;
}
