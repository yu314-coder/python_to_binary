#include <cstdio>
#include <utility>
struct Buf {
    int *p; int n;
    Buf(int c) { n = c; p = new int[c]; }
    Buf(Buf &&o) { p = o.p; n = o.n; o.p = 0; o.n = 0; }
    Buf &operator=(Buf &&o) { p = o.p; n = o.n; o.p = 0; o.n = 0; return *this; }
    int size() const { return n; }
};
int main() { Buf a(5); Buf b(std::move(a)); printf("%d %d\n", a.size(), b.size()); return 0; }
