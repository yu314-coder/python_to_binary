// push_back of a vector of pointers takes `const T &`, a reference to a pointer, and is handed a pointer or an address.
#include <cstdio>
#include <vector>
struct Shape { int n; };
int main() { std::vector<int> a; a.push_back(1); int x = 2; a.push_back(x); Shape s{3}; std::vector<Shape *> b; Shape *p = &s; b.push_back(p); b.push_back(&s); printf("%d %d\n", a[1], b[1]->n); return 0; }
