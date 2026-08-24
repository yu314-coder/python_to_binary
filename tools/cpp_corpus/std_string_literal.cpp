#include <stdio.h>
#include <string>
int main(void) {
    std::string a("hello ");
    std::string b("world");
    std::string c = a + b;
    std::string d("hello world");
    printf("%s|%d|%d\n", c.c_str(), c.size(), c == d);
    return 0;
}
