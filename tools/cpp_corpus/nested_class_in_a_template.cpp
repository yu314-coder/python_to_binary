#include <cstdio>
template <typename T> struct Outer { struct Inner { T v; }; Inner made(T x) { Inner i; i.v = x; return i; } };
int main() { Outer<int> o; printf("%d\n", o.made(6).v); return 0; }
