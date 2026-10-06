// Functions and methods answering `int &` that can throw: read through, assigned through, and thrown out of.
#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
static int arr[4] = {10, 20, 30, 40};
int &get(int i) { if (i > 3) throw E(i); return arr[i]; }
struct Box { int d[4]; int &at(int i) { if (i > 3) throw E(i); return d[i]; } };
int main() {
    int total = 0;
    try { total += get(1); get(2) = 99; int x = get(2) + get(0); total += x; total += get(7); } catch (const E &e) { total += e.c * 1000; }
    Box b; b.d[0] = 1; b.d[1] = 2; b.d[2] = 3; b.d[3] = 4;
    try { b.at(1) += 5; total += b.at(1); total += b.at(9); } catch (const E &e) { total += e.c * 100000; }
    printf("%d %d\n", total, arr[2]);
    return 0;
}
