// A string built from a sum and a counted object inside a try in a loop: the count is right at the handler and after.
#include <cstdio>
#include <string>
static int live = 0;
struct Noisy { Noisy() { live++; } ~Noisy() { live--; } };
struct E { int c; E(int v) : c(v) {} };
int main() {
    Noisy keep;
    for (int i = 0; i < 3; i++) {
        try { std::string s = "item" + std::to_string(i); Noisy n; if (i == 1) throw E(i); printf("%s\n", s.c_str()); }
        catch (const E &e) { printf("caught %d live %d\n", e.c, live); }
    }
    printf("live %d\n", live);
    return 0;
}
