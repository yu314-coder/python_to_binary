#include <stdio.h>
struct V { int kind; union { int i; float f; } value; };
int main(){ V v; v.kind = 1; v.value.i = 7; printf("%d %d\n", v.kind, v.value.i); return 0; }
