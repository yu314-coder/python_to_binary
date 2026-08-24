#include <stdio.h>
class Node { public: int v; Node *next; Node() { v = 0; next = 0; } int get() { return v; } };
int main(void) { Node a; Node b; a.v = 1; b.v = 2; a.next = &b;
  printf("%d %d\n", a.get(), a.next->get()); return 0; }
