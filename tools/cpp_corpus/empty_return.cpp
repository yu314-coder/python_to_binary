#include <cstdio>
#include <string>
static std::string pick(int n) {
    if (n < 0) { return {}; }
    return std::string("ok");
}
class Box {
public:
    Box() : v(0) { }
    Box(int start) : v(start) { }
    int v;
};
static Box choose(int n) {
    if (n == 0) { return {}; }
    return Box(n);
}
int main() {
    printf("[%s] [%s] %d %d\n", pick(-1).c_str(), pick(1).c_str(),
           choose(0).v, choose(6).v);
    return 0;
}
