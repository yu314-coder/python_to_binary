#include <stdio.h>
class Q { public: int thing; int other; Q() { thing = 4; other = 5; }
  int sum() { return thing + other; } };
int main(void) { Q q; printf("%d %d %d\n", q.thing, q.other, q.sum()); return 0; }
