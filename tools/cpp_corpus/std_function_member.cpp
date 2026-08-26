#include <stdio.h>
#include <functional>
class Button { public: std::function<void(int)> on_click;
  void click(int v) { on_click(v); } };
int main(){ Button b; int base = 10;
  b.on_click = [base](int v){ printf("%d\n", base + v); };
  b.click(5); return 0; }
