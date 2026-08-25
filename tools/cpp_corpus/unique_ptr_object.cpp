#include <stdio.h>
#include <memory>
class T { public: int n; T(int v){n=v;} int twice(){return n*2;} };
int main(){ std::unique_ptr<T> p(new T(6)); printf("%d %d\n", p->n, p->twice()); return 0; }
