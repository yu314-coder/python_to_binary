// `S s{};` where the struct holds an array and a struct of its own. Built
// one member at a time - which is for a member whose class has a constructor
// to call - the array was handed `= 0`, which is not C, and the nested
// struct was left out and arrived holding whatever the stack held. A struct
// of plain values is an aggregate in C exactly as it is in C++.
//
// Read member by member rather than byte by byte: C++ says every member is
// zero and says nothing at all about the padding between them.
#include <cstdio>
#include <string>

struct Inner { int a; int b; };

struct Outer {
    short kind;
    Inner inner;
    char pad[8];
    unsigned short port;
};

// And the one this pass is for: a member whose class writes a constructor,
// which the list has to call rather than assign.
struct Named {
    std::string name;
    int count;
};

int main() {
    Outer one{};
    int filled = 0;
    for (int index = 0; index < 8; index++) {
        filled += one.pad[index];
    }
    std::printf("%d %d %d %d %d\n", (int)one.kind, one.inner.a, one.inner.b,
                filled, (int)one.port);
    one.kind = 2;
    one.inner.a = 7;
    one.port = 45454;
    std::printf("%d %d %d %d\n", (int)one.kind, one.inner.a, one.inner.b, (int)one.port);

    Named named = {"socket", 3};
    std::printf("%s %d\n", named.name.c_str(), named.count);
    Named empty{};
    std::printf("[%s] %d\n", empty.name.c_str(), empty.count);
    return 0;
}
