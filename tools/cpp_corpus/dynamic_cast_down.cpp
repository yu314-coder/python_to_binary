#include <stdio.h>
class B { public: virtual ~B(){} virtual int t(){return 1;} };
class D : public B { public: int t(){return 2;} int extra(){return 7;} };
int main(){ B *b = new D(); D *d = dynamic_cast<D*>(b);
  printf("%d %d\n", b->t(), d ? d->extra() : -1); delete b; return 0; }
