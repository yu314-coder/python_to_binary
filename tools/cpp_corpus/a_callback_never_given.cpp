// `if (callback)` on a std::function that nothing in the program ever set.
//
// A class that keeps an optional callback asks whether it holds one before
// calling it. py2bin rewrote that test only while filling the holders of a
// signature some lambda had been assigned to; a signature with no lambda
// anywhere in the program was skipped whole, so `if (onMessage)` reached the
// C as a struct in a condition and was refused. One holder never given
// anything, one given a lambda, and one given a plain function - each tested
// before it is called, and the empty one tested negated too.
#include <stdio.h>
#include <string>
#include <functional>

struct Bridge {
    std::function<void(const std::string &)> onMessage;
    std::function<int(int)> onScale;
    std::function<void(int)> onTick;
    int said = 0;
    void say(const std::string &w) {
        if (onMessage) onMessage(w);
        if (!onMessage) said += (int)w.size();
    }
    int scale(int x) { return onScale ? onScale(x) : x; }
    void tick(int n) { if (onTick) onTick(n); }
};

static void announce(int n) { printf("tick %d ", n); }

int main(void) {
    Bridge b;
    b.say("hello");
    b.say("!");
    printf("%d %d ", b.said, b.scale(4));
    b.onScale = [](int x) { return x * 10; };
    b.onTick = announce;
    b.tick(7);
    printf("%d %d\n", b.scale(4), b.onMessage ? 1 : 0);
    return 0;
}
