#include <cstdio>
static int live = 0;
struct G { G() { ++live; } ~G() { --live; } };
int main() {
    int n = 0;
    for (int i = 0; i < 5; ++i) {
        G g;
        if (i == 3) goto done;
        n += i;
    }
done:
    printf("%d %d\n", n, live);
    return 0;
}
