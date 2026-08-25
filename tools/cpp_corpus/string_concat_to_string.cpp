#include <stdio.h>
#include <string>
int main() {
    std::string a = std::string("ab") + "cd";
    std::string n = std::to_string(42);
    printf("%s|%s\n", a.c_str(), n.c_str());
    return 0;
}
