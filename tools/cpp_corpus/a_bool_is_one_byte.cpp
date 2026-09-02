// C++'s `bool` holds 0 and 1 and is one byte. Written out as `int` it was a
// *signed* type, so `bool flag : 1` held only 0 and -1: a field set to true
// compared unequal to true, and every sizeof was four times what it should be.
#include <cstdio>

struct Flags { bool a : 1; bool b : 1; unsigned rest : 6; };

int main() {
    Flags f;
    f.a = true; f.b = false; f.rest = 9;
    printf("%d %d %d %d %d\n", (int)sizeof(bool), (int)sizeof f,
           f.a == true, f.b == false, f.rest);
    bool grew = f.a;
    printf("%d %d %d\n", (int)grew, grew ? 1 : 0, (int)(grew + grew));
    return 0;
}
