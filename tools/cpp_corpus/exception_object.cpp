#include <stdio.h>
class Err { public: int code; Err() { code = 5; } };
int f(void) { Err e; throw e; }
int main(void) { try { f(); } catch (Err e) { printf("%d\n", e.code); } return 0; }
