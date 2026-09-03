/* `sum(*q)` hands over the object q points at. A by-value parameter is a
   pointer once this is C, so the caller has to hand over an address - and
   `*q` was read as an expression with no address of its own, left exactly as
   written, and a struct reached the C front end where a pointer was wanted.
   `&*q` is the address, the star and the `&` cancelling. */
#include <cstdio>
struct Point { int x; int y; };
class Held {
public:
    int n;
    Held() { n = 5; }
    int get() { return n; }
};
static int sum(Point p) { return p.x + p.y; }
static int twice(Held h) { return h.get() * 2; }
int main() {
    Point p; p.x = 3; p.y = 4;
    Point *q = &p;
    Held h;
    Held *r = &h;
    printf("%d %d %d %d\n", sum(*q), sum(p), twice(*r), twice(h));
    return 0;
}
