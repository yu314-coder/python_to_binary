// `struct Session { int socket = 0; };` - no method in it, so it read as C
// already and went out exactly as written, with the `= 0` still on the
// member. A value on a member is C++ and belongs in the constructor.
#include <cstdio>
#include <string>

struct Session {
    int socket = 7;
    long long seen = 3;
    const char *label = "idle";
};

// A brace on a member is the same thing said another way.
struct Braced {
    int count{5};
    double weight{1.5};
};

// And what is genuinely C stays as it is: an enum written inside a struct
// carries an `=` that has nothing to do with a member.
struct Plain {
    enum Mode { kOpen = 1, kShut = 2 };
    int state;
};

int main() {
    Session one;
    std::printf("%d %lld %s\n", one.socket, one.seen, one.label);
    Session two;
    two.socket = 45454;
    std::printf("%d %lld\n", two.socket, two.seen);

    Braced braced;
    std::printf("%d %.1f\n", braced.count, braced.weight);

    Plain plain;
    plain.state = Plain::kShut;
    std::printf("%d %d\n", plain.state, (int)Plain::kOpen);
    return 0;
}
