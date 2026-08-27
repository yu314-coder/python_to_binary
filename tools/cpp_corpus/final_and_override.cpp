/* `class X final { ... };` reads exactly like `Type name { ... };` - a name,
   a space, a name, a brace - and the pass that rewrites a brace initialiser
   took it for one. The class came out as `X final( ... );`, turned inside
   out, with every member after it lost. */
#include <stdio.h>

struct Pair { int a; int b; };

class Counter final {
public:
    int n;
    Counter() { n = 0; }
    void bump() { n = n + 1; }
    int get() const { return n; }
};

struct Holder final {
    int held;
};

struct Base { virtual int kind() { return 1; } virtual ~Base() {} };
struct Derived final : Base { int kind() override { return 2; } };

int main() {
    Base *b = new Derived();
    Counter c;
    c.bump();
    c.bump();
    Pair p{3, 4};
    Holder h;
    h.held = 5;
    printf("%d %d %d %d %d\n", c.get(), p.a, p.b, h.held, b->kind());
    return 0;
}
