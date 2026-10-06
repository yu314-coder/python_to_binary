// The complement of a cast, `~(unsigned long)15`, inside a method of a class with a destructor: arithmetic, not a destructor call.
#include <cstdio>
static int gone = 0;
struct Room { unsigned long size; ~Room() { gone++; }
    unsigned long rounded() const { return (size + 15) & ~(unsigned long)15; }
    unsigned long low() const { return size & ~(unsigned long)(7); } };
int main() { { Room r{37}; printf("%lu %lu\n", r.rounded(), r.low()); } printf("%d\n", gone); return 0; }
