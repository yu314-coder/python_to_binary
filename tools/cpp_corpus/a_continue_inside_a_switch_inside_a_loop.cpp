// A `continue` written inside a `switch` inside a loop leaves the loop's
// body, and destroys what that body declared above the switch.
//
// py2bin knew a switch as the body a `break` leaves, and treated it as the
// body a `continue` leaves too. So a `continue` in a case ran only the
// destructors of what the switch itself had declared, and the object the
// loop declared each iteration was never taken apart on that path - a leak,
// and a destructor with a side effect that never ran. The destructor here
// counts, and the count after each loop is the output: a for and a while
// with a continue in one case and a break in another, a do-while, a
// continue inside an if inside a case, two objects declared above the
// switch (both die at the continue, in reverse order), and an inner loop
// with its own switch inside an outer loop declaring its own object (the
// continue leaves the inner loop only).
#include <cstdio>

static int destroyed = 0;
static int last = -1;

struct Counted {
    int tag;
    Counted(int t) : tag(t) {}
    ~Counted() { destroyed++; last = tag; }
};

int main() {
    destroyed = 0;
    for (int i = 0; i < 6; i++) {
        Counted c(i);
        switch (i) {
            case 1:
                continue;
            case 3:
                break;
            default:
                printf("for %d ", c.tag);
        }
        printf("after %d ", i);
    }
    printf("| for %d\n", destroyed);

    destroyed = 0;
    int n = 0;
    while (n < 6) {
        Counted w(n);
        n++;
        switch (n) {
            case 2:
                continue;
            case 4:
                break;
            default:
                printf("while %d ", w.tag);
        }
        printf("after %d ", n);
    }
    printf("| while %d\n", destroyed);

    destroyed = 0;
    int k = 0;
    do {
        Counted d(k);
        k++;
        switch (k) {
            case 2:
                continue;
            default:
                printf("do %d ", d.tag);
        }
    } while (k < 4);
    printf("| do %d\n", destroyed);

    destroyed = 0;
    for (int i = 0; i < 4; i++) {
        Counted c(i);
        switch (i) {
            case 2:
                if (c.tag == 2) {
                    continue;
                }
                break;
            default:
                printf("if %d ", c.tag);
        }
    }
    printf("| if %d\n", destroyed);

    destroyed = 0;
    for (int i = 0; i < 3; i++) {
        Counted a(i);
        Counted b(i + 10);
        switch (i) {
            case 1:
                continue;
            default:
                printf("two %d %d ", a.tag, b.tag);
        }
        printf("last %d ", last);
    }
    printf("| two %d last %d\n", destroyed, last);

    destroyed = 0;
    for (int i = 0; i < 3; i++) {
        Counted outer(i);
        for (int j = 0; j < 3; j++) {
            Counted inner(i * 10 + j);
            switch (j) {
                case 1:
                    continue;
                default:
                    printf("nested %d %d ", outer.tag, inner.tag);
            }
        }
        printf("outer %d ", destroyed);
    }
    printf("| nested %d\n", destroyed);
    return 0;
}
