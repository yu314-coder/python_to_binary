#include <stdio.h>
class View {
public:
    int hits;
    View() { hits = 0; }
    const char *page() { return "<script>function go(){ return {a:1}; }</script>"; }
    void hit() { hits = hits + 1; }
    int run() { auto f = [this]() { hit(); }; f(); return hits; }
};
int main() { View v; printf("%d %d\n", v.run(), (int)(v.page()[0] == '<')); return 0; }
