#include <stdio.h>
void shout(int n) { if (n > 3) throw n * 100; printf("shout %d\n", n); }
int depth3(int n) { shout(n); return n; }
int depth2(int n) { return depth3(n) + 1; }
class Guard {
public:
    int id;
    Guard() { id = 0; }
    int check(int v) { if (v == 13) throw 999; return v + 1; }
};
int main(void) {
    Guard g;
    for (int i = 1; i <= 5; i++) {
        try { int r = depth2(i); printf("got %d\n", r); }
        catch (int e) { printf("caught %d\n", e); }
    }
    try { int c = g.check(13); printf("check %d\n", c); }
    catch (int e) { printf("guard %d\n", e); }
    try { throw 5; }
    catch (...) { printf("any\n"); }
    printf("%d\n", g.check(1));
    return 0;
}
