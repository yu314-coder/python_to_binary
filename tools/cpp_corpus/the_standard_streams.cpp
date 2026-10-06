// `puts`, `putchar`, `fputs`, `fprintf` to stdout and stderr, and `fflush`.
#include <cstdio>
int main() { puts("hello"); putchar('x'); putchar('\n'); fputs("to out\n", stdout); fprintf(stdout, "%d\n", 42); fflush(stdout); fprintf(stderr, "err\n"); return 0; }
