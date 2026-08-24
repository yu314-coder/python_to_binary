#include <stdio.h>
namespace lib {
    class M { public: int v; M(); int get(); };
    M::M() { v = 11; }
    int M::get() { return v; }
}
int main(void) { lib::M m; printf("%d\n", m.get()); return 0; }
