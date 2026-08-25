#include <stdio.h>
template<typename T> class Box { public: T v; Box(T x); T get(); };
template<typename T> Box<T>::Box(T x) { v = x; }
template<typename T> T Box<T>::get() { return v; }
int main(){ Box<int> b(9); printf("%d\n", b.get()); return 0; }
