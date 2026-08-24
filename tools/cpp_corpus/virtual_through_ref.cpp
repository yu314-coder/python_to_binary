#include <stdio.h>
class S { public: virtual int id(){return 1;} virtual ~S(){} };
class T : public S { public: int id(){return 2;} };
int ask(S &s){ return s.id(); }
int main(void){ T t; printf("%d\n", ask(t)); return 0; }
