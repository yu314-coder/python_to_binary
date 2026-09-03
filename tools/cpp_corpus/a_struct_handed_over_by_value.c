/* An aggregate passed and answered by value, which is as ordinary in C as
   anything gets and which py2bin's C compiler used to refuse outright. The
   callee writes to its parameter, so this also says the copy is a copy. */
#include <stdio.h>
struct P { int x; int y; };
static struct P add(struct P a, struct P b) {
    a.x = a.x + b.x;
    a.y = a.y + b.y;
    return a;
}
int main(void) {
    struct P a; a.x = 1; a.y = 2;
    struct P b; b.x = 10; b.y = 20;
    struct P c = add(a, b);
    printf("%d %d %d %d\n", c.x, c.y, a.x, a.y);
    return 0;
}
