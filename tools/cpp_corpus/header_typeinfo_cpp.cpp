/* <typeinfo>: `typeid` on an object reached through a base pointer, which is
   the only question C++ answers at run time and the only one py2bin answers
   at all - it reads the table the object carries. */
#include <typeinfo>
#include <cstdio>

struct Base { virtual ~Base() {} };
struct Derived : Base {};
struct Other : Base {};

int main() {
    Derived one;
    Derived two;
    Other three;
    Base *first = &one;
    Base *second = &two;
    Base *third = &three;
    printf("%d\n", typeid(*first) == typeid(*second));
    printf("%d\n", typeid(*first) == typeid(*third));
    printf("%d\n", typeid(*first) != typeid(*third));
    return 0;
}
