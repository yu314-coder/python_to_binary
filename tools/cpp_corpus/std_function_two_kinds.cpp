#include <stdio.h>
#include <functional>
static int triple(int x){ return x*3; }
int main(){ std::function<int(int)> a = triple;
  std::function<int(int)> b = [](int x){ return x + 100; };
  printf("%d %d\n", a(2), b(2)); return 0; }
