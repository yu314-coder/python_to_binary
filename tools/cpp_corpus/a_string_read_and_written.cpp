#include <cstdio>
#include <string>
#include <sstream>
int main() {
    std::ostringstream o;
    o << "n=" << 42 << " f=" << 1.5 << " s=" << std::string("hi");
    std::string got = o.str();
    std::istringstream in("7 8");
    int a = 0, b = 0;
    in >> a >> b;
    printf("%s|%d\n", got.c_str(), a + b);
    return 0;
}
