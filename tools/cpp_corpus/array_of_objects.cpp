#include <stdio.h>
class P { public: int x; P(){x=5;} int get(){return x;} };
int main(void){ P all[4]; int t = 0;
  for (int i = 0; i < 4; i++) { all[i].x = i; t += all[i].get(); }
  printf("%d\n", t); return 0; }
