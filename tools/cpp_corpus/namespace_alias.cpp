#include <stdio.h>
namespace outer { namespace inner { class Thing { public: int v; Thing() { v = 7; } }; } }
namespace shortcut = outer::inner;
int main(void) { shortcut::Thing t; printf("%d\n", t.v); return 0; }
