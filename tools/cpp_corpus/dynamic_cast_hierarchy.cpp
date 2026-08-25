#include <stdio.h>
class B { public: virtual ~B(){} virtual int t(){return 1;} };
class M : public B { public: int t(){return 2;} };
class D : public M { public: int t(){return 3;} };
class O : public B { public: int t(){return 4;} };
int main(){
  B *all[3]; all[0] = new D(); all[1] = new O(); all[2] = new M();
  for (int i = 0; i < 3; i++) {
    M *m = dynamic_cast<M*>(all[i]);
    D *d = dynamic_cast<D*>(all[i]);
    printf("%d%d%d ", all[i]->t(), m ? 1 : 0, d ? 1 : 0);
  }
  printf("\n"); return 0; }
