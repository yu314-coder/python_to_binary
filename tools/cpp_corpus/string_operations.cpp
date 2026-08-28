#include <cstdio>
#include <string>
int main() {
    std::string s = "hello world";
    std::string t = s.substr(6);
    s += "!";
    std::string u = "a" ; u += t;
    printf("%s|%s|%s|%d|%d\n", s.c_str(), t.c_str(), u.c_str(),
           (int)s.size(), (int)(s == "hello world!"));
    printf("%d %d\n", (int)s.find("world"), (int)(t < s));
    return 0;
}
