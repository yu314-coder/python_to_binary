// The same in C++, where the qualifier is written: a local named after the
// function it calls, which is how a program that holds a socket in `socket`
// reaches `::socket`. py2bin takes the qualifier off - every name is global
// by the time the C is written - and the call on the local, which is not a
// thing that can be called, reaches the function.
#include <cstdio>
#include <cstring>

static int measured = 0;

static int measure(const char *text) {
    measured = measured + 1;
    return (int)::strlen(text);
}

int main() {
    int strlen = (int)::strlen("abcdef");
    printf("%d %d %d\n", measure("abcd"), strlen, measured);
    return 0;
}
