#include <cstdio>
#include <string>
int main() {
    std::string s = "abcdef";
    printf("%d %d %s %d\n", (int)s.rfind("c"), (int)s.find("zz"),
           s.substr(2, 3).c_str(), (int)s.compare("abcdef"));
    return 0;
}
