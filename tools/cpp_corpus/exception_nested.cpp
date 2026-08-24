#include <stdio.h>
int inner(int n) { if (n < 0) throw 1; return n; }
int main(void) {
    try {
        try { inner(-1); }
        catch (int e) { printf("inner %d\n", e); throw 2; }
    } catch (int e) { printf("outer %d\n", e); }
    return 0;
}
