#include <stdio.h>
class Doubler { public: int operator()(int x) { return x * 2; } };
int plain(int x) { return x + 1; }
int main(void) {
    Doubler d;
    printf("%d ", d(5));
    int (*fp)(int) = plain;
    printf("%d\n", fp(5));
    return 0;
}
