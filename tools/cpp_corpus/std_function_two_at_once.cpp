#include <stdio.h>
#include <functional>
class Bus { public: std::function<int(int,int)> op;
  int run(int a, int b){ return op(a, b); } };
static int minus(int a, int b){ return a - b; }
int main(){ Bus s; s.op = minus; printf("%d ", s.run(9, 4));
  s.op = [](int a, int b){ return a * b; }; printf("%d\n", s.run(9, 4)); return 0; }
