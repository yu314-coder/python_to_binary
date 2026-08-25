#include <stdio.h>
struct M { unsigned int flag : 1; unsigned int kind : 7; int rest; };
static void set(struct M *m){ m->flag = 1; m->kind = 100; m->rest = 9; }
int main(void){ struct M m; set(&m);
  printf("%u %u %d %u\n", m.flag, m.kind, m.rest, (unsigned)sizeof(struct M)); return 0; }
