// A function that answers a class by value is rewritten to take the hidden
// pointer a caller provides, and the rewrite found the function's head by
// walking back to the last `;` or `}`. A preprocessing directive has neither,
// so an include guard standing above such a function was walked over and
// rewritten away with the head: the `#ifndef` went, the `#endif` stayed, and
// the build stopped on the orphan. Every header that guards itself and
// answers an object - py2bin's own <string> among them - did this.
#include <stdio.h>
#ifndef ANSWERS_A_CLASS
#define ANSWERS_A_CLASS
struct Box { int v; int get() { return v; } };
Box make(int v) { Box b; b.v = v; return b; }
#endif
int main(void) { Box b = make(7); printf("%d\n", b.get()); return 0; }
