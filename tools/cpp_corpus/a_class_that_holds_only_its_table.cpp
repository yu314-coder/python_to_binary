// A class with virtual methods and no data of its own holds the pointer to
// its table and nothing else: eight bytes, which is what C++ says. py2bin
// gave it a byte of filler as well - the filler is there because C has no
// empty struct, and the test for "empty" asked whether there were any data
// members rather than whether anything at all had been written. So the class
// came out sixteen bytes and every class below it was eight bytes too big.
// Nothing failed: py2bin lays out both sides itself and was consistent with
// itself, so the only symptom was `sizeof` answering wrongly - and a struct
// handed to something outside this program being the wrong shape. COM's
// IUnknown is exactly this class.
#include <cstdio>

class Root {
public:
    virtual int who() { return 1; }
    virtual ~Root() {}
};

class Derived : public Root {
public:
    int extra;
    int who() { return 2; }
};

// Still one byte when there is nothing at all - no data, no table.
class Nothing {
public:
    int twice(int n) { return n * 2; }
};

struct Holder {
    Nothing first;
    Nothing second;
    int after;
};

int main() {
    Derived derived;
    derived.extra = 5;
    Root *at = &derived;
    Nothing several[3];
    Holder holder;
    holder.after = 9;
    printf("%d %d %d %d %d %d %d %d\n",
           (int)sizeof(Root), (int)sizeof(Derived), (int)sizeof(Nothing),
           (int)sizeof(several), (int)sizeof(Holder),
           at->who(), derived.extra, holder.first.twice(4) + holder.after);
    return 0;
}
