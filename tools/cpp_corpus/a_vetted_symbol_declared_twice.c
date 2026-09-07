/* The same external symbol declared twice, which C allows outright and which
   a program that includes two headers naming it gets. py2bin's own <chrono>
   binds `QueryPerformanceCounter` to the ABI it emits a call for and
   <windows.h> declares the same entry point; the path a prototype without
   the `extern` keyword takes had always allowed the repeat, and the path an
   `extern` of a vetted symbol takes had not. Both declarations are checked
   against the same table, so a header that disagrees about what the function
   takes is still refused - by the disagreement, and not by having been
   written down twice. */
#include <stdio.h>

extern int getpid(void);
extern int getpid(void);

extern long strlen(const char *s);
extern long strlen(const char *s);

int main(void) {
    printf("%d %d\n", getpid() > 0, (int)strlen("hello"));
    return 0;
}
