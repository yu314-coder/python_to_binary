// A thread started on a member and its object with arguments, and on a lambda with arguments.
//
// py2bin writes the function a platform starts a thread at; it could write
// one for a callable on its own, for a member and its object, and for a
// free function with bound arguments - and refused the two shapes real code
// writes most: `std::thread t(&Bridge::run, this, 80)` and a lambda given
// arguments. The object or the callable is held by address, and the
// arguments are packed beside it.
#include <thread>
#include <cstdio>

struct Bridge {
    int port;
    int seen;
    void run(int extra, int times) { seen = (port + extra) * times; }
    void start() {
        std::thread t(&Bridge::run, this, 80, 2);
        t.join();
    }
};

static int total = 0;
static void work(int a, int b) { total = a + b; }

int main() {
    Bridge b;
    b.port = 8000;
    b.seen = 0;
    b.start();
    int got = 0;
    std::thread first([&got](int a, int b) { got = a * b; }, 6, 7);
    first.join();
    std::thread second(work, 20, 22);
    second.join();
    Bridge other;
    other.port = 1;
    other.seen = 0;
    std::thread third(&Bridge::run, &other, 2, 3);
    third.join();
    printf("%d %d %d %d\n", b.seen, got, total, other.seen);
    return 0;
}
