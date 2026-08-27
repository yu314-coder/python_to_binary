/* A template written inside a template. It cannot be read until the class
   around it has been written out - until then its calls are on objects of a
   type that does not exist - and what it takes is spelled in terms of
   another template, which is the only way `ComPtr<T>::As` says anything
   about what it was handed. */
#include <stdio.h>

template <class T> class Box {
public:
    T held;
    Box() { held = 0; }
};

template <class T> class Holder {
public:
    T value;
    Holder() { value = 0; }
    template <class U> int fill(Box<U> *target) { target->held = (U)value; return 1; }
    template <class U> U scaled(U by) { return (U)(value * by); }
};

int main() {
    Holder<int> whole;
    whole.value = 7;
    Box<int> one;
    whole.fill(&one);

    Holder<double> fractional;
    fractional.value = 1.5;
    Box<double> other;
    fractional.fill(&other);

    printf("%d %.1f %d\n", one.held, other.held, whole.scaled(3));
    return 0;
}
