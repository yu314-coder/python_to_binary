#include <cstdio>
union Both { int i; unsigned char b[4]; };
struct Flags { unsigned a : 1; unsigned b : 3; unsigned c : 4; };
int main() {
    Both u; u.i = 0;
    u.b[0] = 0x21;
    Flags f; f.a = 1; f.b = 5; f.c = 9;
    printf("%d %d %d %d %d\n", u.i, (int)u.b[0], (int)f.a, (int)f.b, (int)f.c);
    return 0;
}
