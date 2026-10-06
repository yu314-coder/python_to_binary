// A try with nothing in it that throws, and a `return` inside it: the objects of both scopes are taken apart in order.
#include <cstdio>
struct Noisy { int id; Noisy(int i) : id(i) {} ~Noisy() { printf("drop %d\n", id); } };
int f(int x) {
    Noisy outer(1);
    try { Noisy inner(2); if (x > 0) return x * 10; }
    catch (...) { return -1; }
    return 0;
}
int main() { printf("%d\n", f(3)); printf("%d\n", f(0)); return 0; }
