#include <stdio.h>
#include <stdlib.h>
typedef long (*Entry)(void *);
extern int pthread_create(void *handle, void *attributes, Entry entry, void *argument);
extern int pthread_join(void *handle, void *answer);

#define EACH 4000
static char *blocks[2][EACH];

static long grab(void *p) {
    long which = (long)p;
    int i = 0;
    while (i < EACH) {
        char *got = (char *)malloc(48);
        got[0] = (char)(which + 1);
        blocks[which][i] = got;
        i = i + 1;
    }
    return 0;
}

int main(void) {
    void *a; void *b;
    pthread_create(&a, 0, grab, (void *)0);
    pthread_create(&b, 0, grab, (void *)1);
    pthread_join(a, 0);
    pthread_join(b, 0);
    /* Every block distinct, and nobody's byte overwritten by anybody else. */
    int clashes = 0; int wrong = 0; int i = 0; int j = 0;
    while (i < EACH) {
        if (blocks[0][i] == 0 || blocks[1][i] == 0) { wrong = wrong + 1; }
        if (blocks[0][i] != 0 && blocks[0][i][0] != 1) { wrong = wrong + 1; }
        if (blocks[1][i] != 0 && blocks[1][i][0] != 2) { wrong = wrong + 1; }
        j = 0;
        while (j < EACH) { if (blocks[0][i] == blocks[1][j]) { clashes = clashes + 1; } j = j + 1; }
        i = i + 1;
    }
    printf("same address handed to both: %d\n", clashes);
    printf("blocks whose byte was not its own: %d\n", wrong);
    return 0;
}
