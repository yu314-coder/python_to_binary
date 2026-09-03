/* Enough arguments in front of it that the aggregate lands in the memory
   argument area rather than a register, in both classes at once - and one
   more function reached through a pointer, where the signature rather than
   the name is what the two sides agree on. */
#include <stdio.h>
struct P { int x; int y; };
struct D { double a; double b; };

static int many(int a, int b, int c, int d, int e, int f, struct P p, int g) {
    return a + b + c + d + e + f + p.x + p.y + g;
}
static double floating(double a, double b, double c, double d, double e,
                       double f, double g, double h, struct D s, double i) {
    return a + b + c + d + e + f + g + h + s.a + s.b + i;
}
static struct P swapped(struct P p, struct P q) {
    struct P r; r.x = p.y + q.x; r.y = q.y + p.x; return r;
}

int main(void) {
    struct P p; p.x = 100; p.y = 200;
    struct D s; s.a = 0.5; s.b = 0.25;
    struct P (*through)(struct P, struct P) = swapped;
    struct P r = through(p, p);
    printf("%d\n", many(1, 2, 3, 4, 5, 6, p, 7));
    printf("%.2f\n", floating(1, 2, 3, 4, 5, 6, 7, 8, s, 9));
    printf("%d %d\n", r.x, r.y);
    return 0;
}
