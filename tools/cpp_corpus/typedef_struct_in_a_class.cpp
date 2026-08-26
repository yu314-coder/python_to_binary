#include <stdio.h>
typedef struct _Point { int x; int y; } Point;
class Holder { public: Point at; Holder(){ at.x = 3; at.y = 4; }
  int sum() const { return at.x + at.y; } };
int main(){ Holder h; Point p; p.x = 1; p.y = 2;
  printf("%d %d\n", h.sum(), p.x + p.y); return 0; }
