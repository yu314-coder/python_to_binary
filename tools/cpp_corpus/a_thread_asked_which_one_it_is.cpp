// `<thread>` had no answer to the two questions every program using one
// asks: which thread is this, and wait a while. The guard on a `join` -
// don't join yourself - is written with both, and py2bin's header had
// neither, so the file stopped at `std::this_thread::get_id()` with the
// namespace still spelled out in the C.
//
// The identity is asked for by name and is not the handle: Windows hands out
// a handle and two handles to one thread are different numbers, so comparing
// them would have said "not me" every time - which is exactly the answer that
// turns the guard into a deadlock. `std::thread::id` is a class of its own
// with a typedef inside `thread` naming it, rather than a class written
// inside `thread`: a class written inside another is lifted out and every
// mention of its short name rewritten, and `id` is a word other programs use.
//
// A thread started on a plain function taking nothing is here too. It read as
// a callable object, so the address of the *function* was passed as though it
// were the address of a variable holding one - and the type that needed had
// no name, because the name of the function is what the typedef is written
// for and this pass had already taken it out of the text.
#include <cstdio>
#include <chrono>
#include <thread>

static int done = 0;
static int counted = 0;

static void work() { done = 1; }

static void add(int a, int b) { counted = a + b; }

int main() {
    std::thread first(work);
    std::thread::id theirs = first.get_id();
    std::thread::id mine = std::this_thread::get_id();
    int same = (mine == theirs) ? 1 : 0;
    int apart = (mine != theirs) ? 1 : 0;
    first.join();

    std::thread second(add, 20, 22);
    second.join();

    // A thread that has finished, and one that never started: neither is a
    // thread, and `id` says so with a zero of its own.
    std::thread never;
    int nobody = (never.get_id() == std::thread::id()) ? 1 : 0;

    // A sleep is a floor: it returns no earlier than it was asked to and the
    // machine decides the rest, so that is what is checked.
    auto begin = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    long long slept = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - begin).count();

    auto until = std::chrono::steady_clock::now();
    until += std::chrono::milliseconds(20);
    std::this_thread::sleep_until(until);
    long long total = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - begin).count();

    // And one already past, which is not a sleep at all.
    std::this_thread::sleep_until(begin);
    std::this_thread::yield();

    printf("%d %d %d %d %d %d %d\n", done, counted, same, apart, nobody,
           (int)(slept >= 28), (int)(total >= 48));
    return 0;
}
