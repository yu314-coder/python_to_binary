#include <stdio.h>
__attribute__((unused)) static int helper(void) { return 1; }
static int __attribute__((unused)) helper2(void) { return 2; }
int __cdecl third(void) { return 3; }
void __stdcall fourth(void) { }
static void copy(char * __restrict d, const char * __restrict s) { *d = *s; }
struct __declspec(novtable) Q { int a; };
int main(void) { char x = 'z', y = 0; struct Q q; q.a = 4; copy(&y, &x); fourth();
  printf("%d %d %d %d %c\n", helper(), helper2(), third(), q.a, y); return 0; }
