#include <cstdio>
template<typename T> struct Box { T v; };
template<typename T> Box<T> wrap(T value) { Box<T> made; made.v = value; return made; }
int main() { Box<int> b = wrap(7); printf("%d\n", b.v); return 0; }
