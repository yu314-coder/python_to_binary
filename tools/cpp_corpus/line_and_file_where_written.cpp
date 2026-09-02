// __LINE__ and __FILE__ answer where they stand in the file the user wrote,
// not where they end up. py2bin pastes every header in above the program and
// rewrites the classes below it before the preprocessor ever runs, so both
// of those move the text a long way from where it started.
#include <cstdio>
#include <cstring>

struct Where {
    int built;
    Where() : built(__LINE__) {}
    int asked() const { return __LINE__; }
};

struct Deeper : Where {
    Deeper() : Where() {}
    int also() const { return __LINE__; }
};

template <typename T>
int in_a_pattern(T) {
    return __LINE__;
}

static const char *just_the_name(const char *path) {
    const char *last = path;
    for (const char *at = path; *at; ++at) {
        if (*at == '/' || *at == '\\') {
            last = at + 1;
        }
    }
    return last;
}

int main() {
    Where w;
    Deeper d;
    printf("built %d\n", w.built);
    printf("asked %d\n", w.asked());
    printf("also %d\n", d.also());
    int whole = in_a_pattern<int>(1);
    int fraction = in_a_pattern<double>(2.5);
    printf("pattern %d %d\n", whole, fraction);
    printf("here %d\n", __LINE__);
    printf("spelled %s\n", "__LINE__ and __FILE__ inside a literal stay put");
    printf("file %s\n", just_the_name(__FILE__));
    printf("same file %d\n", strcmp(just_the_name(__FILE__),
                                    "line_and_file_where_written.cpp") == 0);
    return 0;
}
