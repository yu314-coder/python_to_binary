// <sstream> and nothing else. Every class in it holds a `string`, and the
// header that declares one was not part of it - so `struct ostringstream {
// string held; }` was emitted above the type of its own member and the C
// stage said there was no such type.
#include <sstream>
#include <cstdio>

int main() {
    std::ostringstream out;
    out << "n=" << 42 << ' ' << 1.5;
    std::printf("%s\n", out.str().c_str());

    std::istringstream in(std::string("7 8"));
    int first = 0;
    int second = 0;
    in >> first >> second;
    std::printf("%d\n", first + second);
    return 0;
}
