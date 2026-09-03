// A class body is lifted out of the file before the directives around it are
// read, so an `#if` that picks between two definitions of one class handed
// both of them to the emitter and the C compiler reported `Thing__value`
// defined twice - a mangled name nobody wrote, blaming the wrong thing.
// What is inside an `#if 0` is not part of the program at all, and that is
// the half of this a reader can settle without being the preprocessor: the
// arms no build takes are emptied before anything reads them, so a class in
// one is gone rather than emitted beside the one that replaced it. What is
// left - an `#ifdef` nothing here can answer - is refused by name.
#include <stdio.h>

#if 0
struct Thing { int value() { return 1; } };
#ifdef WHATEVER_THIS_IS
struct Buried { int value() { return 2; } };
#else
struct Buried { int value() { return 3; } };
#endif
#endif

struct Thing { int value() { return 100; } };

#if 1
struct Picked { int value() { return 10; } };
#else
struct Picked { int value() { return 20; } };
#endif

#if 0
struct Second { int value() { return 30; } };
#else
struct Second { int value() { return 40; } };
#endif

#if 0
struct Third { int value() { return 50; } };
#elif 1
struct Third { int value() { return 60; } };
#else
struct Third { int value() { return 70; } };
#endif

#define HAVE_LOG 1

#ifdef HAVE_LOG
struct Logger { int level() { return 3; } };
#endif

// A directive inside a method body is left exactly where it stands, and the
// preprocessor still picks between the two returns.
struct Body {
    int value() {
#ifdef HAVE_LOG
        return 8;
#else
        return 9;
#endif
    }
};

// So is one between the members of a struct that declares no method: its
// body is emitted as it was written, conditionals and all.
struct Plain {
    int a;
#ifdef HAVE_LOG
    int b;
#endif
};

// And a conditional every arm of which can be read leaves nothing behind at
// all - the `#if 0` and its `#endif` go with what they bracket. Left there,
// the `#endif` stood where a member goes and the whole class came out as C
// that will not build.
struct Trimmed {
    int a;
#if 0
    int dead;
    int gone() { return dead; }
#endif
    int live() { return a + 1; }
};

int main(void) {
    Thing t;
    Picked p;
    Second s;
    Third d;
    Body b;
    Plain plain;
    Trimmed trimmed;
    plain.a = 1;
    trimmed.a = 4;
    printf("%d %d %d %d %d\n",
           t.value(), p.value(), s.value(), d.value(), b.value());
    printf("trimmed %d %d\n", trimmed.live(), (int)sizeof(Trimmed));
#ifdef HAVE_LOG
    Logger log;
    printf("log %d\n", log.level());
#else
    printf("no log\n");
#endif
    printf("plain %d\n", (int)sizeof(Plain) + plain.a);
    return 0;
}
