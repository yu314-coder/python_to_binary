#include <cstdio>
#include <string>
static int which(int v) { return 1; }
static int which(double v) { return 2; }
static int which(const char *v) { return 3; }
static int which(const std::string &v) { return 4; }
struct S { int f(int a) { return 10; } int f(int a, int b) { return 20; } };
int main() {
    std::string s = "x";
    S o;
    printf("%d %d %d %d %d %d\n", which(1), which(1.5), which("k"), which(s), o.f(1), o.f(1, 2));
    return 0;
}
