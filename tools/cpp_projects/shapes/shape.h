#ifndef SHAPE_H
#define SHAPE_H
class Shape {
public:
    virtual ~Shape() {}
    virtual int area() const = 0;
    virtual const char *name() const = 0;
};
class Square : public Shape {
public:
    int side;
    Square(int s);
    int area() const;
    const char *name() const;
};
#endif
