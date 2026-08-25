#include <stdio.h>
class G { public: int n; G():n(1){} G(int v):n(v){} ~G(){} };
G plain;
static G stored;
G withArgs(7);
int main(){ printf("%d%d%d\n", plain.n, stored.n, withArgs.n); return 0; }
