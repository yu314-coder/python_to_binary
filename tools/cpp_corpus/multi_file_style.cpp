#include <stdio.h>
class Shape { public: virtual int area() = 0; virtual ~Shape(){} };
class Box : public Shape { public: int w; int h; Box(){w=2;h=3;} int area(){ return w*h; } };
int total(Shape **all, int n){ int t = 0; for (int i = 0; i < n; i++) t += all[i]->area(); return t; }
int main(void){ Box b1; Box b2; b2.w = 4; Shape *all[2]; all[0] = &b1; all[1] = &b2;
  printf("%d\n", total(all, 2)); return 0; }
