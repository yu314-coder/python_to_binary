#include <stdio.h>
template<typename A, typename B>
class Pair { public: A first; B second; Pair(){} };
int main(void){ Pair<int, double> p; p.first = 2; p.second = 1.5;
  printf("%d %.1f\n", p.first, p.second); return 0; }
