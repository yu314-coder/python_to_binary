#include <stdio.h>
#include <string>
class Person { std::string name; int age;
public: Person(std::string n, int a) : name(n), age(a) {}
  std::string label() const { return name + std::string(" is here"); }
  int years() const { return age; } };
int main(){ Person p(std::string("Ada"), 36);
  printf("%s %d\n", p.label().c_str(), p.years()); return 0; }
