#include <stdio.h>
#include <stdarg.h>
static void show(const char *tag, ...){ va_list a; const char *s; va_start(a, tag);
  while ((s = va_arg(a, const char *)) != 0) { printf("%s%s", tag, s); } va_end(a);
  printf("\n"); }
int main(void){ show("|", "a", "bb", "ccc", (const char *)0); return 0; }
