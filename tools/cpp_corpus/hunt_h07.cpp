#include <stdio.h>
class Counter { public: int n; Counter() { n = 0; } void bump(int by = 1) { n += by; } };
int main() { Counter c; c.bump(); c.bump(5); printf("%d\n", c.n); return 0; }
