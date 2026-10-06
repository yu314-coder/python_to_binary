// `Bytes result(MAX_SIZE);` - a number given by a macro, which chooses `vector(n)` and not the copy constructor.
#include <cstdio>
#include <vector>
#include <cstdint>
#ifndef MY_H
#define MY_H
# define MAX_SIZE                 64
#define WIDE 0x10UL
#endif
typedef std::vector<uint8_t> Bytes;
int main() {
    Bytes result(MAX_SIZE);
    std::vector<int> w(WIDE);
    Bytes copy(result);
    printf("%zu %zu %zu\n", result.size(), w.size(), copy.size());
    return 0;
}
