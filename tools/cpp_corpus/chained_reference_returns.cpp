#include <cstdio>
struct Builder {
    int n; char buf[64]; int at;
    Builder() : n(0), at(0) { buf[0] = 0; }
    Builder &add(char c) { buf[at++] = c; buf[at] = 0; n++; return *this; }
    const char *text() const { return buf; }
    int count() const { return n; }
};
int main() {
    Builder b;
    b.add('x').add('y').add('z');
    printf("%s %d\n", b.text(), b.count());
    return 0;
}
