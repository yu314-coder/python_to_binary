#include <stdio.h>
class V { public: int n; V(){n=0;} V(int x){n=x;}
  V operator+(V o){ V r(n+o.n); return r; }
  V operator-(V o){ V r(n-o.n); return r; }
  int operator==(V o){ return n == o.n; }
  int operator<(V o){ return n < o.n; }
  V &operator+=(V o){ n += o.n; return *this; } };
int main(void){ V a(3); V b(4); V c = a + b; V d = b - a;
  a += b;
  printf("%d %d %d %d %d\n", c.n, d.n, a.n, (int)(c == c), (int)(d < c)); return 0; }
