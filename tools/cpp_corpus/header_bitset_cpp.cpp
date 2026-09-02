/* <bitset>: set, counted and tested. */
#include <bitset>
#include <cstdio>

int main() {
    std::bitset<8> bits;
    bits.set(0);
    bits.set(3);
    printf("%d %d\n", (int)bits.count(), (int)bits.size());
    printf("%d %d\n", (int)bits.test(3), (int)bits.test(1));
    bits.reset(3);
    printf("%d\n", (int)bits.count());
    return 0;
}
