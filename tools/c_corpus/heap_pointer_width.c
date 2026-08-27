#include <stdio.h>
#include <stdlib.h>
/* The arena is mapped wherever the kernel likes, which on a 64-bit process
   is usually above four gigabytes. Every pointer must survive the trip
   through the allocator's own arithmetic intact. */
int main(void) {
    unsigned long long a = (unsigned long long)malloc(64);
    unsigned long long b = (unsigned long long)malloc(64);
    int *held = (int *)malloc(sizeof(int) * 4);
    held[0] = 11; held[3] = 44;
    printf("wide=%d gap=%llu high=%d readback=%d %d\n",
           (int)(sizeof(void *) == 8),
           b - a,
           (int)(a > 0xFFFFFFFFULL),
           held[0], held[3]);
    return 0;
}
