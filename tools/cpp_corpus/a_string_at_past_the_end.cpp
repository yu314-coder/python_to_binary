// `string::at` past the end throws `std::out_of_range`; the handler names no parameter.
#include <cstdio>
#include <stdexcept>
#include <string>
int main() {
    std::string s = "hi";
    try { char c = s.at(9); printf("not reached %c\n", c); }
    catch (const std::out_of_range &) { printf("string out of range\n"); }
    return 0;
}
