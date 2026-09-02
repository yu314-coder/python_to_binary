// `sizeof r` where `r` is a reference asks how big the object is. A
// reference to a class is carried as a pointer here and used with `->`,
// which is right for every other use of it and wrong for this one: the
// answer was the pointer's own width, so `memset(&r, 0, sizeof r)` cleared
// eight bytes of an object of whatever size it really was.
#include <cstdio>
struct Odd { char x[7]; };
struct Pair { int a; double b; };
static void by_reference(Odd &o, Pair &p) { printf("%d %d ", (int)sizeof o, (int)sizeof p); }
static void by_pointer(Odd *o) { printf("%d ", (int)sizeof *o); }
template <class T> static void whatever(T &v) { printf("%d ", (int)sizeof v); }
int main() {
    Odd o; Pair p;
    by_reference(o, p);
    by_pointer(&o);
    whatever(o);
    whatever(p);
    printf("%d %d\n", (int)sizeof o, (int)sizeof p);
    return 0;
}
