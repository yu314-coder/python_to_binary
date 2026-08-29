#include <cstdio>
#include <string_view>
int main() { std::string_view s = "hello"; printf("%d %c\n", (int)s.size(), s[1]); return 0; }
