#include <stdio.h>
class C { public: const int limit; int n; C() : limit(10), n(0) { } int room() const { return limit - n; } };
int main(void){ C c; printf("%d\n", c.room()); return 0; }
