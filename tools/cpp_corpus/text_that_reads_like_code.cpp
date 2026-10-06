// Strings holding text that reads like C++. Every pass that turns C++ into C
// read a string as readily as code: "p.twice()" printed `Point__twice(&p)`,
// "Point(1)" built a Point, and "class A { };" printed nothing at all.
#include <cstdio>
#include <string>
#include <vector>
struct Point {
    int x;
    Point() : x(0) { printf("Point()\n"); }
    Point(int x) : x(x) {}
    int twice() const { return x * 2; }
    static Point make() { return Point(7); }
};
int main() {
    Point p(3);
    const char *shapes[] = {
        "Point(1)", "Point{1}", "Point p(1);", "new Point(1)", "Point::make()",
        "p.twice()", "std::vector<Point> v;", "return Point();", "throw Point(1);",
        "template<typename T>", "operator+(", "~Point()", "this->x",
        "[&](int a) { return a; }", "delete p;", "static_cast<int>(x)",
        "class A { };", "struct B : A { };", "namespace n { }", "enum class E { A };",
        "typeid(Point)", "dynamic_cast<Point *>(p)", "a ? b : c", "x << y",
    };
    for (const char *s : shapes) printf("%s\n", s);
    std::string joined = std::string("Point(") + "2" + ")";
    char open = '{', close = '}', paren = '(';
    Point made;
    printf("%s %d %d %c%c%c %d\n", joined.c_str(), p.twice(), Point::make().x,
           open, paren, close, made.x);
    return 0;
}
