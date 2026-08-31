#include <cstdio>
static int live = 0;
struct G { G() { ++live; } ~G() { --live; } };
int main() {
    int n = 0;
    for (int i = 0; i < 5; ++i) {
        G g;
        if (i == 3) { break; }
        n += i;
    }
    int shallow = live;
    for (int i = 0; i < 5; ++i) {
        G g;
        if (i == 2) { G inner; break; }
        n += i;
    }
    printf("%d %d %d\n", n, shallow, live);
    return 0;
}
