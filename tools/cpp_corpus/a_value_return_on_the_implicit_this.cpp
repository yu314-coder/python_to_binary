// A method returning an object by value returns nothing in the C: the caller
// provides the space and the callee writes through a hidden pointer. So the
// call is not an expression anything can be reached on, and the pass that
// writes the temporary out has to see it first. It keyed on a receiver being
// *written*, and C++ lets the receiver of a call on `this` go unwritten - so
// `name().c_str()` inside a method came out as `A__name(this).c_str()`,
// which is not C, and the build stopped with "this expression is not an
// lvalue". `this->name().c_str()` always worked, which is what made it look
// like a problem with value returns rather than with the spelling.
#include <string>
#include <stdio.h>

struct A {
    std::string name() { return std::string("a"); }
    std::string tagged(const char *with) { return std::string(with); }
    void show_inline() { printf("inline %s\n", name().c_str()); }
    void show_argument() { printf("argument %s\n", tagged("b").c_str()); }
    void show_out();
    unsigned long how_long() { return name().size(); }
};

void A::show_out() { printf("out %s\n", name().c_str()); }

struct B : A {
    void show_inherited() { printf("inherited %s\n", name().c_str()); }
};

int main(void) {
    A a;
    a.show_inline();
    a.show_out();
    a.show_argument();
    B b;
    b.show_inherited();
    printf("main %s %lu\n", a.name().c_str(), a.how_long());
    return 0;
}
