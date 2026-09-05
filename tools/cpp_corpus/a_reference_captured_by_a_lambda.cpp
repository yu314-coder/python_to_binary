// A lambda with a capture-default inside a function that was given
// references: `[&]` captures what the body uses and the scope has, and a
// parameter is as much a part of that scope as a local. Read from the body's
// own text alone, neither was captured at all, and the C named both as
// though they were globals - which nothing had declared. A name the lambda
// is given itself is not a capture, whatever the scope calls the same word.
#include <cstdio>
#include <string>
#include <functional>

static int total = 0;
static void run(std::function<void()> f) { f(); }

static void handle(const std::string &message, int &counter, int by) {
    auto post = [&]() { counter = counter + (int)message.size() + by; total = total + 1; };
    run(post);
    post();
}

static void named(const std::string &message, int &seen) {
    // `message` here is the lambda's own, and the enclosing one is not it.
    auto touch = [&](const std::string &message) { seen = seen + (int)message.size(); };
    touch("ab");
    touch(message);
}

int main() {
    int n = 0;
    std::string s = "abcd";
    handle(s, n, 10);
    int seen = 0;
    named("xyz", seen);
    printf("%d %d %d\n", n, total, seen);
    return 0;
}
