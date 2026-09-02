#include <stdio.h>
#include <stddef.h>
struct S { char a; int b; };
struct B { int a; unsigned f : 3; };
struct C2 { unsigned a : 3; unsigned b : 5; };
int main(void) {
    printf("%d %d %d\n", (int)offsetof(struct S, b), (int)sizeof(struct B), (int)sizeof(struct C2));
    return 0;
}
