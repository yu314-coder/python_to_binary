/* Every shape the platform ABIs classify differently - one eightbyte, two,
   all-float, all-double, mixed, and one too big for either - answered by
   value and read straight out of the answer. `make().m` is not an lvalue in
   C, which is why reading it as one refused the whole expression. */
#include <stdio.h>
struct Two { int x; int y; };
struct Wide { long a; long b; };
struct Floats { float x; float y; };
struct Doubles { double a; double b; double c; double d; };
struct Mixed { int n; double d; };
struct Big { int v[16]; };

static struct Two two(int n) { struct Two r; r.x = n; r.y = n * 2; return r; }
static struct Wide wide(void) { struct Wide r; r.a = 5; r.b = 6; return r; }
static struct Floats floats(void) { struct Floats r; r.x = 1.5f; r.y = 2.25f; return r; }
static struct Doubles doubles(void) {
    struct Doubles r; r.a = 1; r.b = 2; r.c = 3; r.d = 4; return r;
}
static struct Mixed mixed(void) { struct Mixed r; r.n = 7; r.d = 0.5; return r; }
static struct Big big(void) {
    struct Big r; int i;
    for (i = 0; i < 16; i++) { r.v[i] = i; }
    return r;
}

int main(void) {
    struct Floats f = floats();
    struct Doubles d = doubles();
    struct Big b = big();
    printf("%d %d\n", two(3).x, two(3).y);
    printf("%ld %ld\n", wide().a, wide().b);
    printf("%.2f %.2f\n", (double)f.x, (double)f.y);
    printf("%.1f %.1f\n", d.a + d.b, d.c + d.d);
    printf("%d %.2f\n", mixed().n, mixed().d);
    printf("%d %d %d\n", b.v[0], b.v[8], b.v[15]);
    printf("%d %d %d %d %d %d\n",
           (int)sizeof(struct Two), (int)sizeof(struct Wide),
           (int)sizeof(struct Floats), (int)sizeof(struct Doubles),
           (int)sizeof(struct Mixed), (int)sizeof(struct Big));
    return 0;
}
