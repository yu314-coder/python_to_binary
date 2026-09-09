// `std::string escapeJson(const std::string &value);` called as
// `escapeJson(state)` where `state` is a `const char *`. C++ builds a string
// and binds the reference to that; py2bin took the address of the pointer and
// handed the C stage a `char **`.
//
// The check that stops it was already there - a reference to a class binds to
// an object of that class and only then - but it looked the parameter's type
// up by where the argument stands in the *call* rather than by where the
// parameter stands in the *declaration*. Those differ by one for every
// function that answers an object, because the caller's space goes in front:
// so for exactly the calls that return something, the check found nothing and
// the address was taken anyway.
#include <cstdio>
#include <string>

static std::string escaped(const std::string &value) {
    std::string out;
    for (unsigned long i = 0; i < value.size(); i++) {
        if (value[i] == '"') out += '.'; else out += value[i];
    }
    return out;
}

// One that answers nothing, where the positions line up and always did.
static int counted(const std::string &value) { return (int)value.size(); }

static void show(const char *state, const std::string &named) {
    printf("%s|%s|%s|%d|%d\n", escaped(state).c_str(), escaped(named).c_str(),
           escaped("li\"teral").c_str(), counted(state), counted(named));
}

int main() {
    show("a\"b", std::string("c\"d"));
    return 0;
}
