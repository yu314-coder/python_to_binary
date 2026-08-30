#include <cstdio>
#include <string>
int main() {
    std::string e;
    std::string s = "hello world hello";
    size_t first = s.find("hello");
    size_t last = s.rfind("hello");
    size_t none = s.find("zzz");
    printf("%d %d %d %d %d %d\n", (int)e.size(), (int)e.empty(),
           (int)first, (int)last, (int)(none == std::string::npos),
           (int)s.substr(6, 5).size());
    return 0;
}
