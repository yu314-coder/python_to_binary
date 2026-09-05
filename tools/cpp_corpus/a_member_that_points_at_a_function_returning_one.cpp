// Data members whose name sits inside parentheses, in every shape a program
// writes: a plain pointer to a function, a qualified one, an array of them, a
// two-word result type, a pointer result, and a pointer to a function that
// answers another pointer. Each has to be a member of the struct, or the
// struct is laid out without it and every offset after it is wrong.
#include <stdio.h>

static int twice(int v) { return v * 2; }
static int thrice(int v) { return v * 3; }
static int (*pick(int which))(int) { return which ? thrice : twice; }
static unsigned long big(int v) { return (unsigned long)v * 1000UL; }
static const char *name(int v) { return v ? "yes" : "no"; }

struct Holder {
    int before;
    int (*op)(int);
    int (* const fixed)(int);
    int (*ops[3])(int);
    unsigned long (*wide)(int);
    const char *(*text)(int);
    int (*(*get)(int))(int);
    int after;
    Holder() : before(1), op(twice), fixed(thrice), wide(big), text(name), get(pick), after(9) {
        ops[0] = twice; ops[1] = thrice; ops[2] = twice;
    }
};

struct Plain { int before; int (*f)(int); int after; };

int main() {
    Holder h;
    printf("%d %d\n", (int)sizeof(Holder), (int)sizeof(Plain));
    printf("%d %d %d %lu %s\n", h.op(5), h.fixed(5), h.ops[1](7), h.wide(4), h.text(1));
    printf("%d %d %d %d\n", h.get(0)(6), h.get(1)(6), h.before, h.after);
    unsigned char raw[96];
    for (int i = 0; i < 96; i++) { raw[i] = (unsigned char)i; }
    Holder *p = (Holder *)raw;
    printf("%d\n", p->after);
    return 0;
}
