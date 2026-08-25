#include <stdio.h>
class Bridge { public: int calls; Bridge(); int run(); void note(); };
Bridge::Bridge() { calls = 0; }
void Bridge::note() { calls = calls + 1; }
int Bridge::run() { auto f = [this]() { note(); }; f(); f(); return calls; }
int main() { Bridge b; printf("%d\n", b.run()); return 0; }
