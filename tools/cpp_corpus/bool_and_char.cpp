#include <stdio.h>
class F { public: int flag; char tag; F() { flag = 1; tag = 'x'; }
  int on() { return flag; } char letter() { return tag; } };
int main(void) { F f; printf("%d %c\n", f.on(), f.letter()); return 0; }
