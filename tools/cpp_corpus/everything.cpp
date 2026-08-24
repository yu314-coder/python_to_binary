#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <string.h>
template<typename T> T twice(T v) { return v + v; }
class Base { public: virtual int tag() { return 1; } virtual ~Base() { } };
class Sub : public Base { public: int tag() { return 2; } };
int risky(int n) { if (n < 0) throw std::runtime_error("no"); return n; }
int main() {
  std::vector<int> v; v.push_back(3); v.push_back(1);
  std::sort(v.begin(), v.end());
  Base *b = new Sub;
  const wchar_t *w = L"wide";
  char buf[8]; strcpy(buf, "hi");
  try { risky(-1); } catch (std::exception &e) { std::cout << e.what() << std::endl; }
  std::cout << v[0] << b->tag() << twice(2) << buf << (int)w[0] << std::endl;
  delete b; return 0;
}
