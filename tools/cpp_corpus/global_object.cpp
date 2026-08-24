#include <stdio.h>
class Counter { public: int n; Counter() { n = 0; } int bump() { n = n + 1; return n; } };
Counter shared;
int use(void) { return shared.bump(); }
int main(void) { shared.bump(); use(); printf("%d %d\n", shared.bump(), shared.n); return 0; }
