#include <stdio.h>
class Box { public: Box(int w) { w_ = w; } int width() const; private: int w_; };
int Box::width() const { return w_; }
int main() { Box b(7); printf("%d\n", b.width()); return 0; }
