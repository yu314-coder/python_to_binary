/* A struct with no method of its own used to be C already and was emitted
   above the classes - which is the wrong side of a class it holds, so
   `struct Config { std::string name; };` never built at all. The struct
   holding nothing but builtins, the one holding another struct, and the two
   that name each other through pointers are here to say that the reading of
   those did not move. */
#include <stdio.h>
#include <string>

struct Res {
    int n;
    ~Res() { printf("~Res %d\n", n); }
};

/* No method of its own, and a class held by value: it has to be emitted
   after `Res`, and its member has to be destroyed with it. */
struct Holder { Res r; };

/* Held through a pointer, which needs no complete type - so this one is
   still plainly C, and `Edge` below is only declared where it is used. */
struct Edge;

struct Node {
    int id;
    Edge *next;
};

struct Edge {
    int weight;
    Node *to;
};

/* Builtins only, a bitfield and an array among them: emitted from the text
   it was written as, which is the only reading either of those survives. */
struct Bits {
    unsigned int low : 3;
    unsigned int high : 5;
    int xs[3];
};

struct Point { int x; int y; };
struct Line { Point a; Point b; };

/* A struct holding a struct holding a class: the middle one stops being
   plainly C, and the outer one has to notice that it now holds a class. */
struct Middle { Res r; };
struct Outer { Middle m; int tag; };

struct Config { std::string name; };

/* An aggregate initialiser is C already, so it is the one declaration form
   nothing rewrites - which is how the object stayed off the list the scope
   destroys on the way out. */
struct Pair { Res r; int tag; };

int main() {
    {
        Holder h;
        h.r.n = 4;
        printf("held %d\n", h.r.n);
    }
    Node one;
    Edge edge;
    one.id = 1;
    one.next = &edge;
    edge.weight = 40;
    edge.to = &one;
    printf("graph %d %d %d\n", one.id, one.next->weight, edge.to->id);
    Bits bits;
    bits.low = 5;
    bits.high = 17;
    bits.xs[0] = 8;
    bits.xs[2] = 9;
    printf("bits %u %u %d %d %d\n", bits.low, bits.high,
           bits.xs[0], bits.xs[2], (int)sizeof(Bits));
    Line line;
    line.a.x = 1; line.a.y = 2; line.b.x = 3; line.b.y = 4;
    printf("line %d %d %d %d %d\n", line.a.x, line.a.y, line.b.x, line.b.y,
           (int)sizeof(Line));
    {
        Outer o;
        o.m.r.n = 7;
        o.tag = 2;
        printf("outer %d %d\n", o.m.r.n, o.tag);
    }
    Config config;
    config.name = "named";
    printf("config %s\n", config.name.c_str());
    {
        Pair pair = { { 3 }, 6 };
        printf("pair %d %d\n", pair.r.n, pair.tag);
    }
    return 0;
}
