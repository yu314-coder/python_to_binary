#include <cstdio>
#include <string>
int main() {
    std::string s;
    for (int i = 0; i < 20; ++i) s += "ab";
    std::string t = s;
    t += "!";
    printf("%d %d %d %d\n", (int)s.size(), (int)t.size(), (int)(s < t), (int)(s == s));
    return 0;
}
