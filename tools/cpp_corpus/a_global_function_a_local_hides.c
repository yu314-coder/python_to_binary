/* `::name(...)` - the global one, said so because a name nearer than it would
   otherwise win: a program that keeps a socket in a variable called `socket`
   and then calls the function of that name has no other way to say which.
   C has no such qualifier and py2bin left it standing, so the C stage met a
   `:` where an expression goes. The qualifier comes off, and a call on a
   local that cannot be called at all - an int is not a function - reaches
   the function of that name, which is the only reading a C compiler would
   accept either. */
#include <stdio.h>
#include <string.h>

static int measured = 0;

static int measure(const char *text) {
    measured = measured + 1;
    return (int)strlen(text);
}

int main(void) {
    int strlen_of_ab = measure("ab");
    printf("%d %d %d\n", strlen_of_ab, measure("abcd"), measured);
    return 0;
}
