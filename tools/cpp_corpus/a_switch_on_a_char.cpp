#include <cstdio>
static int rank(char c) {
    switch (c) {
        case 'a': case 'b': return 1;
        case 'c': { int extra = 2; return extra; }
        case 'd': break;
        default: return -1;
    }
    return 0;
}
int main() { printf("%d %d %d %d %d\n", rank('a'), rank('b'), rank('c'), rank('d'), rank('z')); return 0; }
