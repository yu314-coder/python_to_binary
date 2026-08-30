#include <cstdio>
template <typename T> struct Counter { static int made; Counter() { made++; } };
template <typename T> int Counter<T>::made = 0;
int main() { Counter<int> a, b; Counter<char> c; printf("%d %d\n", Counter<int>::made, Counter<char>::made); return 0; }
