#include <cstdio>
template <typename T> T twice(T v) { return v + v; }
template <> const char *twice<const char *>(const char *v) { return v; }
int main() { printf("%d %s\n", twice(20), twice("kept")); return 0; }
