// A subscript on a member of another object: `b.v[0]`, `p->v[0]`, `b.m["k"]`.
//
// The receiver of `operator[]` is not a variable here but a member reached
// through one. It was left for the C compiler, which read `b.v[0]` as
// pointer arithmetic on a struct and refused. Read, assigned, chained into a
// method call, through an object and through a pointer, for a vector and a
// map, and against a wide string element.
#include <stdio.h>
#include <string>
#include <vector>
#include <map>

struct Bridge {
    std::vector<int> v;
    std::map<std::string, int> counts;
    std::map<std::string, std::wstring> routes;
};

int main(void) {
    Bridge b;
    Bridge *p = &b;
    b.v.push_back(7);
    b.v.push_back(8);
    printf("%d %d\n", b.v[0], p->v[1]);
    b.v[0] = b.v[0] + 1;
    p->v[1] = 20;
    printf("%d %d\n", b.v[0], b.v[1]);
    b.counts["k"] = 5;
    b.counts["k"] += 2;
    printf("%d %d\n", b.counts["k"], (int)b.counts.size());
    b.routes["home"] = L"index.html";
    p->routes["about"] = L"about.html";
    printf("%d %d %d\n", (int)b.routes["home"].size(), (int)p->routes["about"].size(), (int)b.routes.size());
    return 0;
}
