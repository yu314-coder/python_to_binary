// An object declared before a loop, one inside its try: reaching the handler takes apart only the inner one.
#include <cstdio>
struct Noisy { int id; Noisy(int i) : id(i) { printf("make %d\n", id); } ~Noisy() { printf("drop %d\n", id); } };
struct E { int c; };
int main() {
    Noisy outside(0);
    for (int i = 0; i < 3; i++) {
        try { Noisy inside(i + 10); if (i == 1) throw E{i}; printf("body %d\n", i); }
        catch (const E &e) { printf("caught %d\n", e.c); }
    }
    printf("end\n");
    return 0;
}
