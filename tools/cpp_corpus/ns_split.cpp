#include <stdio.h>
namespace app { class A { public: int v; A() { v = 1; } int get() { return v; } }; }
namespace app { int extra(void) { return 7; } }
int main(void) { app::A a; printf("%d %d\n", a.get(), app::extra()); return 0; }
