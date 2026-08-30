#include <iostream>
struct V { int x, y; };
std::ostream &operator<<(std::ostream &o, const V &v) { o << "(" << v.x << "," << v.y << ")"; return o; }
int main() { V v; v.x = 1; v.y = 2; std::cout << v << std::endl; return 0; }
