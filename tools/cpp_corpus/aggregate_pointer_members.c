#include <stdio.h>
struct S { const char *name; int n; };
static struct S table[2] = { {"a", 1}, {"b", 2} };
int main(){ printf("%s%d%s%d\n", table[0].name, table[0].n, table[1].name, table[1].n); return 0; }
