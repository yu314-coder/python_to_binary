#include <stdio.h>
class S { public: virtual int id(){return 1;} virtual ~S(){} };
class T : public S { public: int id(){return 2;} };
int ask(S *s){ return s->id(); }
S *pick(T *t){ return t; }
int main() { T t; S *p = pick(&t); printf("%d %d\n", ask(&t), p->id()); return 0; }
