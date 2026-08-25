#include <stdio.h>
#include <string>
class Bridge {
public:
    std::string name; int calls;
    Bridge() { name.assign("bridge"); calls = 0; }
    const char *label() const { return name.c_str(); }
    void note() { calls = calls + 1; }
    int drive() {
        auto onMessage = [this](const char *what) { note(); printf("%s:%s ", label(), what); };
        onMessage("open");
        onMessage("close");
        return calls;
    }
};
int main() { Bridge b; printf("| %d\n", b.drive()); return 0; }
