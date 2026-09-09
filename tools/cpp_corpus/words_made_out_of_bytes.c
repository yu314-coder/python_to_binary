/* `WSAStartup(MAKEWORD(2, 2), &data)` is the first line of every program
   that opens a socket on Windows, and MAKEWORD is a macro of the platform's
   that py2bin's own <windows.h> did not have - so the call read as one to a
   function nothing declares. These are the set that goes with it.

   <windows.h> on the two Windows targets, and skipped elsewhere the way the
   sweep reads the guard. */
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>

int main(void) {
    WORD version = MAKEWORD(2, 2);
    LONG both = MAKELONG(0x1234, 0x5678);
    printf("%d %d %d %d %d %d\n", (int)version, (int)LOBYTE(version),
           (int)HIBYTE(version), (int)LOWORD(both), (int)HIWORD(both),
           (int)MAKEWORD(1, 0));
    return 0;
}

#else

int main(void) {
    printf("not a Windows target\n");
    return 0;
}

#endif
