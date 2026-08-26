/* COM's IUnknown is exactly three slots. A fourth - a virtual destructor,
   say - would put every derived interface's methods one slot further down
   than the object being called actually has them, and a vtable call is a
   load and a branch, so nothing would report it. */
#include <unknwn.h>
#include <stdio.h>

class IThing : public IUnknown {
public:
    virtual HRESULT DoIt(int n) = 0;
    virtual HRESULT Twice(int n) = 0;
};

class Thing : public IThing {
public:
    unsigned long refs;
    Thing() { refs = 1; }
    HRESULT QueryInterface(REFIID riid, void **o) { *o = (void *)this; return S_OK; }
    unsigned long AddRef() { refs = refs + 1; return refs; }
    unsigned long Release() { refs = refs - 1; return refs; }
    HRESULT DoIt(int n) { return (HRESULT)(n + 1); }
    HRESULT Twice(int n) { return (HRESULT)(n * 2); }
};

/* Reached the way a library reaches it: through the table, at the slot COM
   says the method is at. */
static long through_the_table(IThing *thing) {
    void **table = *(void ***)thing;
    long (*doit)(void *, int) = (long (*)(void *, int))table[3];
    long (*twice)(void *, int) = (long (*)(void *, int))table[4];
    return doit((void *)thing, 10) + twice((void *)thing, 10);
}

int main() {
    Thing t;
    t.AddRef();
    printf("%ld %lu\n", through_the_table(&t), (unsigned long)t.Release());
    return 0;
}
