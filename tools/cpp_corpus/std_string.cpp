#include <stdio.h>
#include <string>
int main(void) {
    std::string a;
    a.assign("hello ");
    std::string b;
    b.assign("world");
    std::string c = a + b;
    std::string d;
    d.assign("hello world");
    printf("%s|%d|%d|%d\n", c.c_str(), c.size(), c == d, a == b);
    return 0;
}
