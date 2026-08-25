#include <stdio.h>
class View {
public:
    int count;
    View() { count = 0; }
    void bump(int n) { count += n; }
    int run() {
        auto handler = [this](int n) { bump(n); count += 1; this->count += 100; };
        handler(5);
        return count;
    }
};
int main() { View v; printf("%d\n", v.run()); return 0; }
