#include <stdio.h>
#include <string>
int main() {
    std::string a("hello");
    a += " world";
    a[0] = 'H';
    printf("%s|%c|%d\n", a.c_str(), a[4], (int)(a < std::string("z")));
    return 0;
}
