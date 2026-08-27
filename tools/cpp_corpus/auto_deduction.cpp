#include <cstdio>
struct Node { int v; const char *label; };
int main() {
    Node n; n.v = 7; n.label = "seven";
    void *raw = &n;
    auto* p = reinterpret_cast<Node*>(raw);
    const auto* q = static_cast<const Node*>(raw);
    auto plain = p->v;
    auto& ref = n.v;
    ref = 9;
    Node **pp = &p;
    auto** deep = pp;
    printf("%d %s %d %d %d\n", p->v, q->label, plain, n.v, (*deep)->v);
    return 0;
}
