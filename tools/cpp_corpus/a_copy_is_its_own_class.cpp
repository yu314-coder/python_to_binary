// A Base copied from a Derived - through a reference, by assignment, or as a
// parameter taken by value - is a Base, and answers virtual calls as one. Its
// bytes copied, it kept the Derived's table and said "derived" four times.
#include <cstdio>
#include <vector>
struct Base { int v = 1; virtual const char *name() const { return "base"; } virtual ~Base() {} };
struct Derived : Base { int extra = 2; const char *name() const override { return "derived"; } };
const char *copy_and_ask(const Base &r) { Base b = r; return b.name(); }
int main() {
    Derived d;
    const Base &r = d;
    Base sliced = r;
    Base assigned; assigned = r;

    printf("%s %s %s %s\n", r.name(), sliced.name(), assigned.name(), copy_and_ask(d));
    return 0;
}
