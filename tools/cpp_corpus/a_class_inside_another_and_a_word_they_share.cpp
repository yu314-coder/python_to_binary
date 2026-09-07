// A class written inside another is lifted out under a name carrying both,
// and every bare mention of the short name is rewritten to follow it. That
// rewrite ran over the whole file: a struct further down with a *member*
// called `tag` had its member renamed too, and the program was told its own
// struct held no such thing. The bare name means the nested class only where
// the outer one's name is in scope - inside its body and inside a method of
// it defined further down - and everywhere else the class is named through
// the outer one, which has been rewritten already.
#include <cstdio>

class Outer {
public:
    class tag {
    public:
        int n;
        tag() { n = 7; }
        int twice() { return n * 2; }
    };
    tag held;
    int total();
    tag make();
};

int Outer::total() {
    tag other;
    other.n = 5;
    return held.n + other.twice();
}

Outer::tag Outer::make() {
    tag made;
    made.n = 11;
    return made;
}

// The same word, for something that has nothing to do with any of it.
struct Record {
    int tag;
    int other;
};

static int tag(int given) { return given + 1; }

int main() {
    Outer outer;
    Outer::tag apart;
    Record record;
    record.tag = 3;
    record.other = 4;
    printf("%d %d %d %d %d %d\n", outer.total(), record.tag, record.other,
           apart.twice(), outer.make().n, tag(8));
    return 0;
}
