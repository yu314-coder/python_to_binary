#include <cstdio>
struct S { char c; double d; };
int main() { printf("%d %d\n", (int)alignof(double), (int)sizeof(S)); return 0; }
