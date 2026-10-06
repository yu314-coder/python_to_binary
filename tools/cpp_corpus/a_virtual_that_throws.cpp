// A virtual function that throws, reached through an array of base pointers given derived ones in its list.
#include <cstdio>
struct E { int c; E(int v) : c(v) {} };
struct Shape { virtual ~Shape() {} virtual int area() const = 0; };
struct Bad : Shape { int area() const override { throw E(13); } };
struct Good : Shape { int area() const override { return 4; } };
int main() {
    Good g; Bad b; Shape *all[] = {&g, &b, &g};
    int sum = 0;
    for (Shape *s : all) { try { sum += s->area(); } catch (const E &e) { sum += e.c * 100; } }
    printf("%d\n", sum);
    return 0;
}
