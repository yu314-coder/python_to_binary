#include <cstdio>
struct Reg {
    static int made;
    enum Kind { small = 1, big = 2 };
    Kind k;
    Reg(Kind kind) : k(kind) { made++; }
    static int howMany() { return made; }
};
int Reg::made = 0;
int main() {
    Reg a(Reg::small), b(Reg::big);
    printf("%d %d %d %d\n", (int)a.k, (int)b.k, Reg::howMany(), Reg::made);
    return 0;
}
