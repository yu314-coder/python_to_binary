#include <cstdio>
static_assert(sizeof(int) == 4, "int is four bytes");
template <typename T> struct Box { static_assert(sizeof(T) > 0, "no"); T v; };
int main() { Box<int> b; b.v = 5; printf("%d\n", b.v); return 0; }
