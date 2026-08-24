#include <stdio.h>
int fib(int n){ return n < 2 ? n : fib(n-1) + fib(n-2); }
class T { public: int depth(int n){ return n == 0 ? 0 : 1 + depth(n-1); } };
int main(void){ T t; printf("%d %d\n", fib(10), t.depth(5)); return 0; }
