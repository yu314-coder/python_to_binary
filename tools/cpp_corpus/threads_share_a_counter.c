#include <stdio.h>
typedef long (*Entry)(void *);
extern int pthread_create(void *handle, void *attributes, Entry entry, void *argument);
extern int pthread_join(void *handle, void *answer);
static long total = 0;
static long worker(void *p) {
    long i = 0;
    while (i < 100000) { __py2bin_atomic_add(&total, 1); i = i + 1; }
    return 0;
}
int main(void) {
    void *a; void *b;
    int made_a = pthread_create(&a, 0, worker, 0);
    int made_b = pthread_create(&b, 0, worker, 0);
    pthread_join(a, 0);
    pthread_join(b, 0);
    printf("%d %d %ld\n", made_a, made_b, total);
    return 0;
}
