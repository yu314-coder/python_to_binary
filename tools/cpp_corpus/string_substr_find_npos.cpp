#include <stdio.h>
#include <string>
int main() {
    std::string a("hello world");
    std::string b = a.substr(6, 5);
    printf("%s|%d|%d\n", b.c_str(), (int)a.find("world"), (int)(a.find("zzz") == std::string::npos));
    return 0;
}
