#include <cstdio>
/* The PIMPL spelling: the name is declared, the body is somewhere this file
   is not compiled with. Everything is done through the pointer, so C++ asks
   for nothing more than the name - and C needs to be told the name is a type
   or `Session *inside;` is a declaration of nothing. */
class Session;
struct Ticket;
struct Door { Session *inside; Ticket *stub; int tag; };
Session *as_session(void *raw) { return (Session *) raw; }
int main() {
    int store = 41;
    Door d;
    d.inside = as_session(&store);
    d.stub = 0;
    d.tag = 7;
    printf("%d %d %d %d\n", d.tag, *(int *) d.inside,
           (int)sizeof(Door), d.stub == 0);
    return 0;
}
