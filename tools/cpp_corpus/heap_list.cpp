#include <stdio.h>
class Cell {
public:
    int value;
    Cell *next;
    Cell(int v) { value = v; next = 0; }
};
class List {
public:
    Cell *head;
    int count;
    List() { head = 0; count = 0; }
    void push(int v) {
        Cell *made = new Cell(v);
        made->next = head;
        head = made;
        count = count + 1;
    }
    int total() {
        int sum = 0;
        Cell *walk = head;
        while (walk != 0) { sum = sum + walk->value; walk = walk->next; }
        return sum;
    }
    void release() {
        Cell *walk = head;
        while (walk != 0) { Cell *after = walk->next; delete walk; walk = after; }
        head = 0;
    }
};
int main(void) {
    List l;
    for (int i = 1; i <= 50; i++) l.push(i);
    printf("%d %d %d\n", l.count, l.total(), l.head->value);
    l.release();
    printf("%d\n", l.head == 0);
    return 0;
}
