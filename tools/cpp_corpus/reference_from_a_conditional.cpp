#include <cstdio>
#include <string>
int main() {
    std::string a = "yes"; std::string b = "no";
    bool pick = true;
    const std::string &r = pick ? a : b;
    printf("%s\n", r.c_str());
    return 0;
}
