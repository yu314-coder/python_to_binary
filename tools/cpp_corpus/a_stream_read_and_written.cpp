#include <cstdio>
#include <sstream>
#include <string>
int main() {
    std::stringstream ss;
    ss << "12 34";
    int a = 0, b = 0;
    ss >> a >> b;
    printf("%d\n", a + b);
    return 0;
}
