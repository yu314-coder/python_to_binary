#include <stdio.h>
#include <functional>
static int base(int x){ return x - 1; }
int main(){ std::function<int(int)> f = base;
  f = [](int x){ return x * 2; };
  printf("%d\n", f(21)); return 0; }
