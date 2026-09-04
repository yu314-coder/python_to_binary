// `typedef class Counter Counter;` - a class named before it is defined.
//
// A generated COM header writes every coclass this way (`typedef class
// DOMDocument DOMDocument;` in <msxml.h>), and C++ allows it before or after
// the definition. C has no `class`, and the C stage met the keyword. It is
// a struct typedef now, which is what the class becomes.
#include <stdio.h>

typedef class Counter Counter;
typedef class Pair Pair;

class Counter {
public:
    int n;
    int bump() { return ++n; }
};

class Pair {
public:
    Counter a;
    Counter b;
    int total() { return a.n + b.n; }
};

typedef class Counter CounterAgain;

int main(void) {
    Counter c;
    c.n = 1;
    Pair p;
    p.a.n = 2;
    p.b.n = 3;
    CounterAgain d;
    d.n = 10;
    printf("%d %d %d\n", c.bump(), p.total(), d.bump());
    return 0;
}
