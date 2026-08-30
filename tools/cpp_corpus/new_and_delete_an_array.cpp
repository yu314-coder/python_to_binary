#include <cstdio>
static int live = 0;
struct N { int v; N() : v(1) { ++live; } ~N() { --live; } };
int main() {
    N *many = new N[4];
    many[2].v = 9;
    int held = live;
    int got = many[2].v + many[0].v;
    delete[] many;
    N *one = new N();
    int after = live;
    delete one;
    printf("%d %d %d %d\n", held, got, after, live);
    return 0;
}
