// compare answers the difference of the first bytes that differ, read unsigned, or one either way where one string runs out: what libc++ answers.
#include <cstdio>
#include <string>
int main() {
    std::string a = "abc", b = "abd", c = "ab", d = "abc";
    printf("%d %d %d %d\n", a.compare(b), a.compare(c), a.compare(d), a.compare("abz"));
    printf("%d %d %d %d %d %d\n", (int)(a == d), (int)(a != b), (int)(a < b), (int)(b > a), (int)(c <= a), (int)(a >= c));
    printf("%d %d\n", (int)(a == "abc"), (int)(a != "xyz"));
    return 0;
}
