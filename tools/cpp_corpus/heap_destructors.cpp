#include <stdio.h>
int alive = 0;
class Tracked {
public:
    int id;
    Tracked() { id = 0; alive = alive + 1; }
    ~Tracked() { alive = alive - 1; }
};
int main(void) {
    Tracked *many = new Tracked[7];
    for (int i = 0; i < 7; i++) many[i].id = i;
    int a = alive;
    delete[] many;
    Tracked *one = new Tracked;
    int b = alive;
    delete one;
    printf("%d %d %d\n", a, b, alive);
    return 0;
}
