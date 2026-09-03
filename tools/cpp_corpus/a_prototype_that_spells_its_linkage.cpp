/* `extern` on a file-scope function names the linkage it has anyway, so a
   prototype carrying it is a forward declaration and nothing more: the body
   arrives further down this same file. py2bin used to read the keyword as a
   claim about its vetted adapter ABI and refuse any name that was not in that
   table, which put every hand-written `extern int helper(int);` out of reach -
   and that is how a great many people write a header.

   A body may also arrive right there, which `extern int twice(int)` below
   does, and the declaration is then the definition. */
#include <cstdio>

extern int helper(int);
extern long long widen(int, int);
extern void announce(const char *what);
extern double scaled(double v);
extern int twice(int x) { return x * 2; }

struct Box { int held; };

extern int unbox(struct Box *box);

int main() {
    struct Box box;
    announce("start");
    box.held = 7;
    std::printf("%d %lld %.2f %d %d\n", helper(20), widen(3, 4), scaled(2.5),
                twice(21), unbox(&box));
    return 0;
}

int helper(int x) { return x * 2 + 1; }

long long widen(int a, int b) { return (long long) a * 1000000000LL + b; }

void announce(const char *what) { std::printf("[%s]\n", what); }

double scaled(double v) { return v * 1.5; }

int unbox(struct Box *box) { return box->held * 3; }
