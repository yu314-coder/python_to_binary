#include <stdio.h>
class Shape { public: virtual ~Shape(){} virtual double area() const = 0; };
class Circle : public Shape { public: double r; Circle(double v):r(v){} double area() const { return 3.14159 * r * r; } };
class Square : public Shape { public: double s; Square(double v):s(v){} double area() const { return s * s; } };
int main(){ Shape *all[2]; all[0] = new Circle(1.0); all[1] = new Square(2.0);
  double t = 0; for (int i = 0; i < 2; i++) t += all[i]->area();
  printf("%.3f\n", t); for (int i = 0; i < 2; i++) delete all[i]; return 0; }
