#include <stdio.h>
struct S { char name[8]; int n; };
int main(){ struct S s = {"hi", 5}; printf("%s%d\n", s.name, s.n); return 0; }
