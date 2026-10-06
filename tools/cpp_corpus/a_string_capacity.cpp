// capacity() says what libc++ says: twenty-two before any storage is taken, then one short of a multiple of eight, twice as much each time.
#include <cstdio>
#include <string>
int main() {
    std::string g; printf("%zu ", g.capacity());
    for (int i = 0; i < 100; i++) { g.push_back('x'); if (i == 22 || i == 23 || i == 47 || i == 48 || i == 99) printf("%zu ", g.capacity()); }
    std::string r; r.reserve(100); printf("%zu\n", r.capacity());
    return 0;
}
