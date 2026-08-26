#include <stdio.h>
#include <string>

int main() {
    std::string a = "abc";
    std::string b = a;
    int n = 5;
    printf("%s %s %d %d\n", a.c_str(), b.c_str(), (int)a.size(), n);
    return 0;
}
