#include <stdio.h>
/* A published C runtime writes the calling convention between the star and
   the name, which is not where a specifier stands - py2bin read the word as
   the name being declared and the real name as a syntax error. A media
   header names its formats with a character constant holding four of them,
   whose value C leaves to the implementation and every implementation
   computes the same way: each character into the next byte down. */
void * __cdecl give(void *p) { return p; }
char * __stdcall next(char *s) { return s + 1; }
const char * __cdecl unchanged(const char *s);
const char * __cdecl unchanged(const char *s) { return s; }

int main(void) {
    int n = 41;
    unsigned int code = 'MJPG';
    printf("%d %c %u %d %s\n", *(int *)give(&n), *next("hi"), code,
           (int)'ab', unchanged("kept"));
    return 0;
}
