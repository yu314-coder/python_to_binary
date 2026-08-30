#include <cstdio>
int main() {
    int n = 0, i = 0;
    do { n += i; ++i; } while (i < 4);
    int k = 0;
    switch (n) { case 5: k = 1; case 6: k += 10; break; case 7: k = 100; break; default: k = -1; }
    int j = 0;
    for (int a = 0; a < 3; ++a) { for (int b = 0; b < 3; ++b) { if (b == 1) continue; if (a == 2) goto done; ++j; } }
done:
    int t = (n > 3) ? ((k > 5) ? 1 : 2) : 3;
    printf("%d %d %d %d\n", n, k, j, t);
    return 0;
}
