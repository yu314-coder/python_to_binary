#include <cstdio>
struct Reg {
    static int total;
    static Reg &one() { static Reg held; return held; }
    int n;
    Reg() : n(0) { ++total; }
    void bump() { ++n; }
    static int howMany() { return total; }
};
int Reg::total = 0;
int main() {
    Reg::one().bump();
    Reg::one().bump();
    Reg a;
    printf("%d %d %d\n", Reg::one().n, Reg::howMany(), a.n);
    return 0;
}
