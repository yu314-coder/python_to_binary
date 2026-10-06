// A vector copied - by a declaration, an assignment, a parameter, a return
// or a struct holding one - is a vector of its own. They shared storage, and
// writing to the copy wrote to the original.
#include <cstdio>
#include <vector>
#include <string>
void bump(std::vector<int> v) { v[0] = 99; }
std::vector<int> grow(std::vector<int> v) { v.push_back(4); return v; }
struct Holder { std::vector<int> items; };
int main() {
    std::vector<int> a{1, 2, 3};
    std::vector<int> b = a;
    b[0] = 10;
    std::vector<int> c; c = a; c[1] = 20;
    bump(a);
    std::vector<int> d = grow(a);
    Holder h1; h1.items = a; Holder h2 = h1; h2.items[2] = 30;
    std::vector<std::vector<int>> grid; grid.push_back(a); grid.push_back(b);
    std::vector<std::vector<int>> copy = grid; copy[0][0] = 77;
    printf("%d %d %d | %d %d | %zu %zu | %d %d | %d %d\n", a[0], a[1], a[2], b[0], c[1],
           a.size(), d.size(), h1.items[2], h2.items[2], grid[0][0], copy[0][0]);
    return 0;
}
