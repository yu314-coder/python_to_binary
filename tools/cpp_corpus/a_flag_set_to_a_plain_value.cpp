// `std::atomic<bool> running_; running_ = false;` - the plainest line there
// is, and py2bin had two things wrong with it.
//
// A class that says how to be assigned *from this value* uses that. `atomic`
// declares `operator=(T)`, and py2bin built an `atomic<bool>` from the `false`
// first and then assigned that object over the member - which handed the
// operator an object where it wanted a value.
//
// And the pass that writes the operator call matched only a *name* on the
// right. `false` is a `0` by the time it reads the statement, so the line was
// left alone and the C stage was handed a struct being assigned an int. The
// address is taken only where the operator wants an object: one taking a plain
// value wants the value.
#include <cstdio>
#include <atomic>
#include <string>

class Host {
public:
    std::atomic<bool> running_;
    std::atomic<int> served_;
    std::string name_;
    Host() { running_ = false; served_ = 0; name_ = "host"; }
    void go() { running_ = true; served_ = 7; }
    void stop() { running_ = false; }
    // `served_.load()` and not `served_`: an object standing where a number
    // goes is a conversion of its own, and this one is about assignment.
    int state() { return (running_ ? 1 : 0) + served_.load(); }
};

int main() {
    Host host;
    int before = host.state();
    host.go();
    int during = host.state();
    host.stop();
    // And from outside the class, which is the same statement.
    host.served_ = 2;
    printf("%d %d %d %s\n", before, during, host.state(), host.name_.c_str());
    return 0;
}
