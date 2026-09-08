// `if (file)` asks the object, and C has no way to ask a struct anything.
// C++ says a condition converts what it is given to bool, and for an object
// that is the class's own conversion - the same one `while (in >> n)` uses
// once the stream has answered. py2bin asked it only for a call to one of the
// class's operators; an object standing in a condition by itself was handed
// to the C stage as a struct, which refused it. Every place a condition is
// asked: an `if`, a `while`, in front of a `?`, and either side of `&&` and
// `||`. `!object` is left where the class writes `operator!`, which is the
// other way a holder says there is nothing here.
#include <cstdio>

class Flag {
public:
    int held;
    Flag(int given) { held = given; }
    operator bool() const { return held != 0; }
    int operator!() const { return held == 0; }
};

// One with no `operator!` of its own, so the conversion answers that too.
class Count {
public:
    int n;
    Count(int given) { n = given; }
    operator bool() const { return n > 0; }
};

// A name in a condition is not always being asked whether it is true: an
// arrow after it means it is being reached through, and a lookahead that
// excluded `.` but not `->` turned `&& this->held` into a conversion called
// on `this` and then reached through.
class Holder {
public:
    Count inside;
    Holder() : inside(3) {}
    int both() { return (inside && this->inside.n > 1) ? 9 : 0; }
};

int main() {
    Flag on(1);
    Flag off(0);
    Count some(2);
    Count none(0);

    int a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0;
    if (on) a = 1;
    if (!off) b = 1;
    c = on ? 2 : 3;
    while (off) { break; }
    d = (on && !off) ? 4 : 5;
    if (some) e = 6;
    if (!none) f = 7;
    if (none || some) g = 8;

    Holder holder;
    printf("%d %d %d %d %d %d %d %d\n", a, b, c, d, e, f, g, holder.both());
    return 0;
}
