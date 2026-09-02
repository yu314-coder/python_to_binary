#include <cstdio>
struct E { };
struct P { char a; int b; };
int main() { E e; (void)e; printf("%d %d\n", (int)sizeof(E), (int)sizeof(P)); return 0; }
