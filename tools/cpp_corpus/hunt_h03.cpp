#include <stdio.h>
class Item { public: int n; Item() { n = 5; } int get() const { return n; } };
int main() {
    Item a; Item *all[1]; all[0] = &a;
    for (int i = 0; i < 1; i++) { printf("%d\n", all[i]->get()); }
    return 0;
}
