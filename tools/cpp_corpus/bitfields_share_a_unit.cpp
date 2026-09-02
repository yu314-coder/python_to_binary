// Bitfields of different declared types share one storage unit: C only asks
// that a field not cross a boundary of its own type, not that its neighbour
// be spelled the same way. Read as "a new unit whenever the declared size
// differs", every struct here came out too big, and each field still read
// back what was written to it - so only the bytes say what went wrong.
#include <cstdio>
#include <cstring>

struct Mixed { unsigned char a : 3; unsigned int b : 5; };
struct Other { unsigned int a : 3; unsigned char b : 5; };
struct Wide  { unsigned char a : 3; unsigned int b : 30; };
struct After { unsigned char a : 3; unsigned int b : 5; char c; };
struct Zero  { unsigned a : 3; unsigned : 0; unsigned b : 3; };
struct Runs  { unsigned short a : 5; unsigned short b : 5; unsigned short c : 5; };

static void show(const char *name, void *at, unsigned bytes) {
    unsigned char *p = (unsigned char *)at;
    printf("%s %d ", name, (int)bytes);
    for (unsigned i = 0; i < bytes; i++) printf("%02x", p[i]);
    printf("\n");
}

int main() {
    Mixed m; memset(&m, 0, sizeof m); m.a = 1; m.b = 2; show("mixed", &m, sizeof m);
    Other o; memset(&o, 0, sizeof o); o.a = 1; o.b = 2; show("other", &o, sizeof o);
    Wide  w; memset(&w, 0, sizeof w); w.a = 1; w.b = 2; show("wide", &w, sizeof w);
    After f; memset(&f, 0, sizeof f); f.a = 1; f.b = 2; f.c = 3; show("after", &f, sizeof f);
    Zero  z; memset(&z, 0, sizeof z); z.a = 1; z.b = 2; show("zero", &z, sizeof z);
    Runs  r; memset(&r, 0, sizeof r); r.a = 1; r.b = 2; r.c = 3; show("runs", &r, sizeof r);
    printf("%d %d %d %d\n", m.a, m.b, w.b, r.c);
    return 0;
}
