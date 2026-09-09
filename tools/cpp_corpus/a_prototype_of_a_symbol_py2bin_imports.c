/* A platform header declares its entry points as plain prototypes - `int
   WSAStartup(WORD, LPWSADATA);` - with no `extern` in front of them. py2bin
   turned the `extern` spelling into an import and left this one as a
   function declared and never defined, so a program calling one was told to
   name a library for something that ships with the system.

   Checked against the same table the `extern` spelling is, so a header that
   disagrees about what the function takes is still refused - by the
   disagreement, and not by having been written without a keyword. */
#include <stdio.h>

int getpid(void);
long strlen(const char *s);

int main(void) {
    printf("%d %d\n", getpid() > 0, (int)strlen("abcd"));
    return 0;
}
