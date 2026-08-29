#include <cstdio>
static int safe() noexcept { return 4; }
struct S { void go() noexcept {} };
int main() { S s; s.go(); printf("%d\n", safe()); return 0; }
