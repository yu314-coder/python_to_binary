#include <stdio.h>
class M { public: enum Mode { Off, On }; Mode m; M() { m = On; } int get() { return (int)m; } };
int main() { M x; printf("%d %d\n", x.get(), (int)M::Off); return 0; }
