#include <stdio.h>
#include <functional>
static void twice(std::function<void(int)> cb){ cb(1); cb(2); }
int main(){ twice([](int v){ printf("%d ", v); }); printf("\n"); return 0; }
