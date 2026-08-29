#include <cstdio>
struct Base { int v; Base() { v = 5; } virtual int get() { return v; } };
struct Only : virtual Base { int extra; Only() { extra = 2; } int both() { return get() + extra; } };
int main() { Only o; printf("%d %d\n", o.get(), o.both()); return 0; }
