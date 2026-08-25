#include <stdio.h>
class L { public: int n; L():n(0){} };
int main(){ static L held; held.n = held.n + 5; printf("%d\n", held.n); return 0; }
