#include <stdio.h>
int main(){ auto outer = [](int a){ return [a](int b){ return a + b; }; };
  auto add5 = outer(5); printf("%d\n", add5(3)); return 0; }
