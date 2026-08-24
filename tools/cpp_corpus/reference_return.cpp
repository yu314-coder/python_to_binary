#include <stdio.h>
class Row {
public:
    int cells[8];
    Row() { for (int i = 0; i < 8; i++) cells[i] = 0; }
    int &at(int i) { return cells[i]; }
    int &operator[](int i) { return cells[i]; }
};
int main(void) {
    Row r;
    r.at(2) = 42;
    r[5] = 7;
    r[5] = r[5] + 1;
    printf("%d %d %d\n", r.at(2), r[5], r.cells[0]);
    return 0;
}
