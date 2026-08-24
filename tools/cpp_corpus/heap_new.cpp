#include <stdio.h>
class Node {
public:
    int value;
    Node() { value = -1; }
    Node(int v) { value = v; }
    ~Node() { }
    int get() { return value; }
};
int main(void) {
    Node *a = new Node(7);
    Node *b = new Node;
    Node *many = new Node[5];
    for (int i = 0; i < 5; i++) many[i].value = i * 3;
    int total = 0;
    for (int i = 0; i < 5; i++) total += many[i].get();
    int *raw = new int[10];
    raw[9] = 55;
    printf("%d %d %d %d\n", a->get(), b->get(), total, raw[9]);
    delete a; delete b; delete[] many;
    return 0;
}
