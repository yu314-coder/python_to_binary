// Two shapes of brace list py2bin did not read. `State s = {"a", 5, 0};` -
// a declaration written with the `=` C has always written one with, which
// the pass that reads brace initialisers only matched without. And
// `at->states[key] = {...};` - an assignment, where the braces mean the type
// on the *left* and C has no brace list in an expression at all. The second
// is built into an object a declaration can hold and then assigned, which is
// what the C++ says one step later.
//
// Underneath both: C++ copy-initialises each member from the value written
// for it, so a `std::string` member takes the constructor a `const char *`
// chooses. Written out as a C initialiser list the string's own struct was
// handed a pointer. Members the list does not reach are value-initialised -
// the default constructor for a class, a zero for anything else - and a
// member of a multi-word type is one member: `long long size;` was read as a
// member called `long`, so the class was said not to have a `size` at all.
#include <cstdio>
#include <map>
#include <string>

struct State {
    std::string name;
    long long size;
    long long offset;
};

struct Holder {
    std::map<std::string, State> states;
};

int main() {
    State written = {"first", 5, 1};
    // Short of its members: the rest are value-initialised.
    State partly = {"second"};
    // And a plain aggregate of numbers, which is C already and stays so.
    struct Pair { int a; int b; } pair = {3, 4};

    Holder holder;
    Holder *at = &holder;
    at->states["one"] = {"third", 7, 2};
    State plain;
    plain = {"fourth", 9, 3};

    State &got = at->states["one"];
    printf("%s %d %d|%s %d %d|%d %d|%s %d %d|%s %d %d\n",
           written.name.c_str(), (int)written.size, (int)written.offset,
           partly.name.c_str(), (int)partly.size, (int)partly.offset,
           pair.a, pair.b,
           got.name.c_str(), (int)got.size, (int)got.offset,
           plain.name.c_str(), (int)plain.size, (int)plain.offset);
    return 0;
}
