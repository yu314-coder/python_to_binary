// Numbers into strings and strings into numbers: leading space skipped, a sign, digits as far as they go.
#include <cstdio>
#include <string>
int main() {
    std::string a = std::to_string(42) + "/" + std::to_string(-7) + "/" + std::to_string(123456789012LL);
    printf("%s %d %ld %lld %.2f %lu\n", a.c_str(), std::stoi("  -123abc"), std::stol("+99"), std::stoll("123456789012"), std::stod("2.5e1"), std::stoul("77"));
    return 0;
}
