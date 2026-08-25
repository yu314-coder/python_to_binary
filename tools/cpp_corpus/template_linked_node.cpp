#include <stdio.h>
template<typename T> class Node { public: T value; Node<T> *next;
  Node(T v): value(v), next(0) {} };
int main(){ Node<int> a(1); Node<int> b(2); a.next = &b;
  printf("%d %d\n", a.value, a.next->value); return 0; }
