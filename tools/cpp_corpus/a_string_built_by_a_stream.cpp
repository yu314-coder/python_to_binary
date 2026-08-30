#include <cstdio>
#include <string>
#include <sstream>
int main() {
    std::ostringstream o;
    o << "n=" << 42 << " f=" << 1.5 << " s=" << std::string("hi");
    std::string got = o.str();
    printf("%s|%d\n", got.c_str(), (int)got.size());
    return 0;
}
