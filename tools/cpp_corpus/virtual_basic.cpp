#include <stdio.h>
class Shape {
public:
    int size;
    Shape() { size = 2; }
    virtual int area() { return 0; }
    virtual const char *name() { return "shape"; }
    int twice() { return area() * 2; }
};
class Square : public Shape {
public:
    Square() { size = 4; }
    int area() { return size * size; }
    const char *name() { return "square"; }
};
class Circle : public Shape {
public:
    Circle() { size = 3; }
    int area() { return size * size * 3; }
};
int main(void) {
    Square s; Circle c; Shape p;
    Shape *all[3]; all[0] = &s; all[1] = &c; all[2] = &p;
    for (int i = 0; i < 3; i++)
        printf("%s %d %d\n", all[i]->name(), all[i]->area(), all[i]->twice());
    return 0;
}
