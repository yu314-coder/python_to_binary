#include <stdio.h>
namespace app {
    class Engine { public: int p; Engine() { p = 10; } int rate() { return p; } };
    class Car { public: Engine motor; int w; Car() { w = 4; } int total() { return motor.rate() + w; } };
    int bonus(void) { return 1; }
}
using namespace app;
int main(void) { Car c; Engine e; printf("%d %d %d\n", c.total(), e.rate(), bonus()); return 0; }
