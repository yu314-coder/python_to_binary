// A guarded header reached directly and again through a header that includes it.
//
// Both headers branch on __cplusplus, so each is preprocessed alone before
// it is pasted - and the run for the second expanded the first inside it,
// knowing nothing of the copy already in the unit. An include of winsock2.h
// followed by an include of ws2tcpip.h (which includes the first) is the
// shape, and "duplicate enumerator 'COMP_EQUAL'" was the result.
#include <stdio.h>
#include <ws.h>
#include <ws2.h>

int main(void) {
    WSAECOMPARATOR c = COMP_EQUAL;
    printf("%d %d %d\n", (int)c, compare_kind(0), address_kind(1));
    return 0;
}
