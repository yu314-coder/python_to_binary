#include <stdio.h>
class G { public: int n; G():n(42){} };
static G global;
int main(){ printf("%d\n", global.n); return 0; }
