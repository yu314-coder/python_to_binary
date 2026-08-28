#include <cstdio>
#include <string>
static const std::string &pick(const std::string &a, const std::string &b) {
    return a.size() >= b.size() ? a : b;
}
int main() { std::string x = "ab", y = "cdef";
    printf("%s\n", pick(x, y).c_str()); return 0; }
