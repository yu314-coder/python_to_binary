// `session->transfers[id]` where `session` is a `shared_ptr<Session>`. A
// smart pointer is a class held by value whose only member is a raw pointer,
// so the table of what an object holds said `session` had no `transfers` -
// and what a program names is on the other side of `operator->`. Two passes
// went wrong for want of it: nothing could say what `session->transfers[id]`
// *was*, so a brace list assigned to it had no type to be built as; and the
// pass that turns a subscript into a call finds its receiver by the name it
// is reached by, so the C compiler was handed `[...]` on a struct and read
// it as pointer arithmetic.
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct State {
    std::string name;
    long long size;
    long long offset;
};

struct Session {
    std::map<std::string, State> transfers;
    std::vector<int> marks;
    State current;
};

int main() {
    auto session = std::make_shared<Session>();
    std::string id = "one";

    session->transfers[id] = {"first", 5, 0};
    session->transfers["two"].size = 11;
    session->marks.push_back(3);
    session->marks.push_back(4);
    session->current.size = 7;

    State &got = session->transfers[id];
    State &other = session->transfers["two"];

    // And through a `unique_ptr`, which reaches the same way.
    std::unique_ptr<Session> alone(new Session());
    alone->transfers["k"].size = 21;
    State &third = alone->transfers["k"];

    printf("%s %d %d|%d|%d %d %d|%d|%d\n", got.name.c_str(), (int)got.size,
           (int)got.offset, (int)other.size, (int)session->marks.size(),
           session->marks[0], session->marks[1], (int)session->current.size,
           (int)third.size);
    return 0;
}
