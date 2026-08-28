#include <cstdio>
#include <bitset>
int main() { std::bitset<8> b(5); printf("%d %s\n", (int)b.count(), b.to_string().c_str()); return 0; }
