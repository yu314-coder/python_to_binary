// Prose inside an `#if 0`, holding one apostrophe that nothing pairs with -
// the way a header explains what it has switched off. A reader that opened a
// character constant on it ran on across the `#endif`, and everything below
// went dark. Comments hold them too: it's fine, isn't it.
#include <stdio.h>

#if 0
This block doesn't compile, and isn't meant to.
#include <no_such_header.h>
struct Counter { int n; int bump() { return n - 1; } };
#endif

/* A block comment doesn't close one either,
   and a "quote inside one is text. */

struct Counter {
    int n;
    int bump() { n += 1; return n; }
};

int main(void) {
    Counter c;
    c.n = 4;
    printf("%d\n", c.bump());
    printf("%c%c\n", '\'', '"');
    printf("%s\n", "it's \"quoted\"");
    return 0;
}
