// The other order: the including header first, then the one it included.
//
// The run that pasted the first header expanded the second inside it; a
// direct include of the second afterwards has to know that, or the unit
// holds it twice.
#include <stdio.h>
#include <ws2.h>
#include <ws.h>

int main(void) {
    WSAECOMPARATOR c = COMP_NOTLESS;
    printf("%d %d %d\n", (int)c, compare_kind(0), address_kind(1));
    return 0;
}
