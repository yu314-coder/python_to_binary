#include <stdio.h>
void swap(int &a, int &b) { int t = a; a = b; b = t; }
void scale(double &v, double by) { v = v * by; }
int main(void) {
    int x = 3, y = 8;
    swap(x, y);
    double d = 2.5;
    scale(d, 4.0);
    int flags = 0xF0;
    int mask = 0x30;
    int both = flags & mask;
    int *heap = new int[4];
    heap[2] = 77;
    printf("%d %d %.2f %d %d\n", x, y, d, both, heap[2]);
    delete[] heap;
    return 0;
}
