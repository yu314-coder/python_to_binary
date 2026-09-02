/* `if (p) { p->method(); }` - a pointer checked and then called through, in
   the same statement pair. The parentheses around a name are taken off where
   they say nothing, and an `if`'s were being taken off with them, so this
   came out as `if p { ... }` and would not compile. It is the commonest
   thing anyone writes with a pointer, and it took ComPtr in <wrl.h> with it:
   every one of its methods is guarded this way. */
#include <cstdio>

struct Counted {
    unsigned long refs;
    Counted() { refs = 1; }
    virtual unsigned long AddRef() { refs = refs + 1; return refs; }
    virtual unsigned long Release() { refs = refs - 1; return refs; }
};

template <class T> class Holder {
public:
    T *ptr_;
    Holder() { ptr_ = 0; }
    Holder(T *given) { ptr_ = given; if (ptr_) { ptr_->AddRef(); } }
    ~Holder() { if (ptr_) { ptr_->Release(); ptr_ = 0; } }
    T *Get() { return ptr_; }
};

int main() {
    Counted thing;
    Counted *plain = &thing;
    if (plain) { printf("%lu\n", plain->AddRef()); }
    while (plain) { printf("%lu\n", plain->Release()); plain = 0; }
    {
        Holder<Counted> held(&thing);
        printf("%lu\n", held.Get()->refs);
    }
    printf("%lu\n", thing.refs);
    return 0;
}
