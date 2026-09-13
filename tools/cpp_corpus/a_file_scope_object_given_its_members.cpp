/* `static Identifier one = {11, 2};` outside every function. A file-scope
   object's constructor has nowhere to run in C - C++ runs it before `main`,
   so this translator writes the call at the top of `main` instead, and takes
   the value off the declaration to pass to it. A plain struct has no
   constructor: it is an aggregate, and writing the members out is what builds
   it, which is C already. Taken off anyway, the value went to a call that was
   never written - and the program read zeroes where it had spelled numbers.
   No diagnostic and no build failure; just the wrong answer.

   The object with a constructor is here too, because that is the case the
   taking-off is for and it has to keep working. */
#include <cstdio>

struct Plain { unsigned long first; int second; };

class Counted {
public:
    Counted(int start) : held(start * 3) {}
    int held;
};

static Plain kept = {11, 2};
Plain shared = {33, 4};
static Counted counted(7);

int main() {
    Plain local = {55, 6};
    std::printf("%lu %d %lu %d %lu %d %d\n", kept.first, kept.second,
                shared.first, shared.second, local.first, local.second,
                counted.held);
    return 0;
}
