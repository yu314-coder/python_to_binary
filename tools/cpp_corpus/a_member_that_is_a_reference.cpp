// `WindowsTransport &transport_;` held by a session, and `transport_.start()`
// inside its methods. C has no reference, so py2bin holds one as a pointer -
// and it was then counted among the pointers, so every call on it was looked
// for as `transport_->start()`, found nowhere, and reached the C stage as a
// member call on a struct that has only data.
//
// A reference is written the way an object held by value is written, with a
// dot, and its address is `&this->transport_` - which the dereference every
// mention of it gets, when the method is written out, turns back into the
// pointer it holds. So it belongs with the members held by value, not with
// the pointers.
#include <cstdio>
#include <string>
#include <vector>

class Engine {
public:
    int runs;
    std::vector<int> marks;
    Engine() { runs = 0; }
    void start() { runs = runs + 1; }
    int count() const { return runs; }
    std::string name() const { return std::string("engine"); }
};

class Session {
public:
    Engine &engine_;
    int seen;
    Session(Engine &engine) : engine_(engine) { seen = 0; }
    void begin() {
        engine_.start();
        seen = engine_.runs;
    }
    // A method answering an object, which is written through a hidden
    // pointer and so asks the receiver for its address a different way.
    std::string called() { return engine_.name(); }
    int asked() { return engine_.count(); }
    // And a container reached through the reference.
    void mark(int one) { engine_.marks.push_back(one); }
    int marked() { return (int)engine_.marks.size(); }
};

int main() {
    Engine engine;
    Session session(engine);
    session.begin();
    session.begin();
    session.mark(7);
    session.mark(8);
    printf("%d %d %s %d %d %d\n", engine.runs, session.seen,
           session.called().c_str(), session.asked(), session.marked(),
           engine.marks[1]);
    return 0;
}
