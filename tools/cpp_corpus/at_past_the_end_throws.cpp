// `vector::at` past the end throws `std::out_of_range`, and a handler for one catches it.
#include <cstdio>
#include <stdexcept>
#include <vector>
int main() {
    std::vector<int> v(3, 1);
    try { int x = v.at(5); printf("not reached %d\n", x); }
    catch (const std::out_of_range &e) { printf("out of range\n"); }
    printf("%d\n", v.at(2));
    return 0;
}
