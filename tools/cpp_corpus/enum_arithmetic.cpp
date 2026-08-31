#include <cstdio>
enum Flag { None = 0, Read = 1, Write = 2, Both = 3 };
enum class Mode { Off, On };
static int combine(Flag a, Flag b) { return (int)a | (int)b; }
int main() {
    Flag f = Read;
    int n = f + 1;
    int both = combine(Read, Write);
    Mode m = Mode::On;
    printf("%d %d %d %d %d\n", (int)f, n, both, (int)(both == Both), (int)m);
    return 0;
}
