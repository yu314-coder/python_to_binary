// The text was C++, and its conditionals were written for a C++ compiler -
// where `__cplusplus` is defined. The C run that follows the translator left
// it undefined, so a program's own `#ifdef __cplusplus` took the arm meant
// for C: the class in the first arm had already been lifted out and written
// as a struct, then the `typedef int Tag;` of the `#else` arm was read too,
// and the build stopped on a name that was "already a different type". A
// function defined under such a guard simply went missing at its call.
#include <stdio.h>
#ifdef __cplusplus
struct Tag { int which() { return 1; } };
#define KIND "cpp"
#else
typedef int Tag;
#define KIND "c"
#endif
#ifdef __cplusplus
static int answer(Tag &t) { return t.which() + 1; }
#endif
int main(void) {
#ifdef __cplusplus
    Tag t; printf("%s %d %d\n", KIND, t.which(), answer(t));
#else
    printf("%s\n", KIND);
#endif
    return 0;
}
