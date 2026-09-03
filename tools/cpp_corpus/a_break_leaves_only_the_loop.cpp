// `break` and `continue` leave the loop, and destroy what the loop declared.
//
// An object declared before a loop outlives every `break` inside it; py2bin
// ran the destructors of everything in the function at each `break` and
// `continue`, as if they were a `return`. A vector declared before a loop
// was freed by the first `break`, and its size read 0 afterwards - silently.
// Objects announce their construction and destruction, so the order is the
// output: a for with a break, a while with a continue, a switch with breaks,
// a break out of a nested block, an object declared inside the loop (which
// the break does destroy), and a return from inside a loop (which destroys
// everything, in reverse order).
#include <stdio.h>
#include <string>
#include <vector>

struct Loud {
    int id;
    Loud(int i) : id(i) { printf("+%d ", id); }
    ~Loud() { printf("-%d ", id); }
};

struct Bridge {
    std::vector<std::wstring> log;
    void say(const std::wstring &w) { log.push_back(w); }
};

static int early(int limit) {
    Loud a(1);
    for (int i = 0; i < 10; ++i) {
        Loud b(2);
        if (i == limit) return i;
    }
    return -1;
}

int main(void) {
    Bridge bridge;
    bridge.say(L"x");
    Loud outer(10);
    for (int i = 0; i < 5; ++i) {
        if (i == 2) break;
    }
    printf("| %d ", (int)bridge.log.size());
    int i = 0;
    while (i < 6) {
        ++i;
        if (i % 2) continue;
        bridge.say(L"y");
    }
    printf("| %d ", (int)bridge.log.size());
    int k = 2;
    switch (k) {
        case 1: printf("one "); break;
        case 2: { Loud inner(20); printf("two "); break; }
        default: printf("other "); break;
    }
    printf("| %d ", (int)bridge.log.size());
    for (int j = 0; j < 3; ++j) {
        Loud each(30 + j);
        { if (j == 1) break; }
    }
    printf("| %d ", (int)bridge.log.size());
    printf("| %d ", early(3));
    printf("| %d\n", (int)bridge.log.size());
    return 0;
}
