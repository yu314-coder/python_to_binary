#include <cstdio>
int main() {
    unsigned char b = 250;
    b = (unsigned char)(b + 10);
    signed char s = -1;
    unsigned u = 3;
    int i = -1;
    long shifted = 1L << 40;
    unsigned short w = 65535;
    w = (unsigned short)(w + 2);
    printf("%d %d %d %ld %d %d\n", (int)b, (int)s, (int)(i < (int)u), shifted, (int)w, (int)(0xFFu >> 4));
    return 0;
}
