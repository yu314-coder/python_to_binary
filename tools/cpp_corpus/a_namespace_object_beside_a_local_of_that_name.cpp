#include <cstdio>
/* A namespace's own variable is a name the flattening has to watch, because
   two of them become one. What a function inside the namespace calls its
   locals is nobody else's business, and counting those as the namespace's
   would refuse this - which C++ builds without a murmur. */
namespace left {
    int count = 5;
    int doubled() { int count = 2; return count * 3; }
}
namespace right {
    int total = 7;
    int trebled() { int count = 4; return count * 10; }
}
int main() {
    left::count += 1;
    right::total += 2;
    printf("%d %d %d %d\n", left::count, right::total,
           left::doubled(), right::trebled());
    return 0;
}
