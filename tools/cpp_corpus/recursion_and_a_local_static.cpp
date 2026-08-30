#include <cstdio>
struct Counter { int n; Counter(int start) : n(start) {} int next() { return ++n; } };
static int step() { static Counter c(10); return c.next(); }
static int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
static int isOdd(int n);
static int isEven(int n) { return n == 0 ? 1 : isOdd(n - 1); }
static int isOdd(int n) { return n == 0 ? 0 : isEven(n - 1); }
int main() { printf("%d %d %d %d %d\n", step(), step(), fib(10), isEven(8), isOdd(7)); return 0; }
