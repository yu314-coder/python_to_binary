/* <sstream>: written into and read out of. */
#include <sstream>
#include <string>
#include <cstdio>

int main() {
    std::ostringstream out;
    out << "n=" << 42 << " f=" << 1.5;
    printf("%s\n", out.str().c_str());
    std::istringstream in("7 8");
    int first = 0;
    int second = 0;
    in >> first >> second;
    printf("%d %d\n", first, second);
    return 0;
}
