#include <stdio.h>
int alive = 0;
class Res {
public:
    int id;
    Res() { alive = alive + 1; id = alive; }
    ~Res() { alive = alive - 1; }
};
int risky(int n) { Res local; if (n < 0) throw 1; return n; }
int main(void) {
    try { int v = risky(-1); printf("%d\n", v); }
    catch (int e) { printf("caught, alive=%d\n", alive); }
    printf("end alive=%d\n", alive);
    return 0;
}
