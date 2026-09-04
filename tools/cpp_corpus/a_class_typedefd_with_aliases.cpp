// `typedef struct Counter { ... methods ... } Counter_t, *PCounter;` - a class
// with typedef names.
//
// The class is lifted out of the text to be emitted with the others, and what
// stayed behind was `typedef Counter_t, *PCounter;` - the aliases with nothing
// in front of them. The lift now leaves `typedef struct Counter Counter_t,
// *PCounter;`, which is the C for what was written.
#include <stdio.h>

typedef struct Counter {
    int n;
    void bump() { n += 1; }
    int twice() const { return n * 2; }
} Counter_t, *PCounter;

typedef struct Pair {
    Counter_t a;
    Counter_t b;
    int total() const { return a.n + b.n; }
} Pair_t;

int main(void) {
    Counter_t c;
    c.n = 4;
    c.bump();
    PCounter p = &c;
    p->bump();
    Pair_t pair;
    pair.a.n = 1;
    pair.b.n = 2;
    printf("%d %d %d\n", c.twice(), p->n, pair.total());
    return 0;
}
