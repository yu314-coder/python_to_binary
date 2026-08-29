#include <stdio.h>
struct S { char c; double d; };
struct T { char a; char b; };
int main(void) {
    printf("%d %d %d %d %d\n", (int)_Alignof(double), (int)_Alignof(char),
           (int)_Alignof(struct S), (int)_Alignof(struct T), (int)sizeof(struct S));
    return 0;
}
