/* <string>: building one, growing it, and the three questions asked of it. */
#include <string>
#include <cstdio>

int main() {
    std::string held = "hello";
    held += ", world";
    printf("%s %d\n", held.c_str(), (int)held.size());
    printf("%d\n", (int)held.find("world"));
    printf("%s\n", held.substr(0, 5).c_str());
    printf("%d %d\n", (int)held.empty(), held[1] == 'e');
    return 0;
}
