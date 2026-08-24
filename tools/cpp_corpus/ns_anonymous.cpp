#include <stdio.h>
namespace {
    class Hidden { public: int v; Hidden() { v = 5; } int get() { return v; } };
    int secret(void) { return 9; }
}
int main(void) { Hidden h; printf("%d %d\n", h.get(), secret()); return 0; }
