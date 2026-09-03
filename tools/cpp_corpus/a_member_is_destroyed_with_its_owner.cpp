// A destructor C++ writes and nobody else does. A class holding another has
// an implicit destructor whose whole job is to take the member apart, and
// py2bin emitted none at all unless somebody wrote `~Box` by hand - so a
// member holding a count, a buffer or a handle was never released. The object
// left scope and its member's destructor simply never ran, which is the kind
// of wrong that builds, runs, and prints a number that is merely incorrect.
//
// The order matters as much as the fact: members come apart in reverse of the
// order they were declared, and a member goes before the base.
#include <stdio.h>

static int alive = 0;
static char order[64];
static int at = 0;

struct A { ~A() { alive--; order[at++] = 'A'; order[at] = 0; } };
struct B { ~B() { alive--; order[at++] = 'B'; order[at] = 0; } };
struct Two { A a; B b; int k() { return 1; } };
struct Nest { Two t; int k() { return 2; } };
struct Base { ~Base() { alive--; order[at++] = 'S'; order[at] = 0; } };
struct Derived : Base { A a; int k() { return 3; } };

struct Res { Res() { alive++; } ~Res() { alive--; } };
struct Box { Res r; int peek() { return 5; } };

int main(void) {
    alive = 2; at = 0; order[0] = 0; { Two t; t.k(); }     printf("two %d [%s]\n", alive, order);
    alive = 2; at = 0; order[0] = 0; { Nest n; n.k(); }    printf("nest %d [%s]\n", alive, order);
    alive = 2; at = 0; order[0] = 0; { Derived d; d.k(); } printf("derived %d [%s]\n", alive, order);
    alive = 0; { Box b; printf("held %d %d\n", alive, b.peek()); }
    printf("released %d\n", alive);
    return 0;
}
