#include <stdio.h>
class L { public: static const int limit = 10; int n; L() { n = 0; } int room() { return limit - n; } };
int main() { L l; printf("%d %d\n", l.room(), L::limit); return 0; }
