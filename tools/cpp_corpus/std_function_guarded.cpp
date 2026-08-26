#include <stdio.h>
#include <functional>
class Emitter { public: std::function<void(int)> handler;
  void send(int v) { if (handler) { handler(v); } else { printf("none "); } } };
int main(){ Emitter e; e.send(1);
  e.handler = [](int v){ printf("got %d ", v); };
  e.send(2); printf("\n"); return 0; }
