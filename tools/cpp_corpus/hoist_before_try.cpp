#include <stdio.h>
int risky(int n) { if (n < 0) throw 1; return n; }
int main(void) {
    printf("one|");
    try { risky(-1); } catch (int e) { printf("caught|"); }
    printf("%d\n", risky(7));
    return 0;
}
