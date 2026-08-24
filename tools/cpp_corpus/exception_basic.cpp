#include <stdio.h>
int risky(int n) { if (n < 0) throw 7; return n * 2; }
int main(void) {
    try { int v = risky(5); printf("ok %d\n", v); }
    catch (int e) { printf("caught %d\n", e); }
    try { int v = risky(-1); printf("ok %d\n", v); }
    catch (int e) { printf("caught %d\n", e); }
    return 0;
}
