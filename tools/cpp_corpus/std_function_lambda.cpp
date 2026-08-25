#include <stdio.h>
#include <functional>
int main(){ std::function<int(int)> f = [](int x){ return x*3; }; printf("%d\n", f(4)); return 0; }
