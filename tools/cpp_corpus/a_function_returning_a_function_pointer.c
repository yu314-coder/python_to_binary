/* A function whose result is a pointer to a function, written with the
   declarator C has for it and no typedef: `int (*pick(int which))(int)`.
   OpenSSL declares every accessor of a method table this way -
   `int (*BIO_meth_get_write(const BIO_METHOD *biom))(BIO *, const char *,
   int);` - and py2bin used to refuse the shape by name, asking for a typedef
   the header did not have. A prototype first, as a header writes it, then
   the definition; the result is called at once, called through `*`, kept in
   a local and in a global, and handed back to the function that made it. */
#include <stdio.h>
static int twice(int x) { return 2 * x; }
static int thrice(int x) { return 3 * x; }
int (*pick(int which))(int);
int (*pick(int which))(int) { return which ? thrice : twice; }
int (*kept)(int);
int main(void) {
    int (*f)(int) = pick(1);
    kept = pick(0);
    printf("%d %d %d %d %d\n", pick(0)(5), (*pick(1))(5), f(7), kept(f(1)), pick(kept(0))(4));
    return 0;
}
