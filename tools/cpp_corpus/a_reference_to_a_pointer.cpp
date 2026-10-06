// A reference to a pointer, given the pointer, an address, or a new object: the reference binds to the pointer, or to a temporary holding the value.
#include <cstdio>
struct Shape { int n; };
int take_ref(const int &v) { return v; }
int take_ptr_ref(Shape *const &p) { return p->n; }
int take_ptr_ref2(const Shape * const &p) { return p->n; }
template <typename T> int take_t(const T &v) { return (int)sizeof(v); }
int main() {
    Shape s{7};
    Shape *p = &s;
    printf("%d %d %d %d %d\n", take_ref(3), take_ptr_ref(p), take_ptr_ref(&s), take_ptr_ref2(new Shape()), take_t<Shape *>(p));
    return 0;
}
