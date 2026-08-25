#include <stdio.h>
template<typename T, int N> class Fixed { public: T items[N]; int size(){ return N; } };
int main(){ Fixed<int, 4> f; f.items[0] = 9; printf("%d %d\n", f.size(), f.items[0]); return 0; }
