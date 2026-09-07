// A thread started with an argument of a class whose only constructor takes
// something. The arguments are put in a small class of py2bin's own so the
// platform can be handed one pointer, and that class assigned its members
// where it should have built them: a member of a class that is never built
// empty cannot be brought into existence and filled in afterwards, and the
// build stopped on a class the program never default-builds. The pack is
// written in the form the initialiser-list pass leaves behind - that pass
// has run by the time the pack is written, so a list written here would
// still be a list when the class is read.
#include <cstdio>
#include <string>
#include <thread>

class Session {
public:
    int id;
    std::string name;
    Session(int given, const char *called) { id = given; name = called; }
    int size() { return id + (int)name.size(); }
};

static int total = 0;
static int seen = 0;

static void handle(Session session, int extra) {
    total = session.size() + extra;
}

class Worker {
public:
    int done;
    Worker() { done = 0; }
    void carry(Session session, int by) { done = session.id * by; seen = seen + 1; }
};

int main() {
    Session one(5, "abc");
    std::thread first(handle, one, 3);
    first.join();

    Worker worker;
    Session two(4, "de");
    std::thread second(&Worker::carry, &worker, two, 10);
    second.join();

    printf("%d %d %d\n", total, worker.done, seen);
    return 0;
}
