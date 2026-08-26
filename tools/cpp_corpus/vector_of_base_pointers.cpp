#include <stdio.h>
#include <vector>

struct Shape {
    virtual ~Shape() {}
    virtual int sides() const = 0;
};
struct Square : Shape { int sides() const { return 4; } };
struct Triangle : Shape { int sides() const { return 3; } };

int main() {
    std::vector<Shape *> all;
    all.push_back(new Square());
    all.push_back(new Triangle());
    int total = 0;
    for (unsigned i = 0; i < all.size(); i++) { total += all[i]->sides(); }
    printf("%d %d\n", (int)all.size(), total);
    return 0;
}
