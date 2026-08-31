#include <cstdio>
struct W {
    virtual int nine(int a, int b, int c, int d, int e, int f, int g, int h, int i) {
        return a * 100000000 + b * 10000000 + c * 1000000 + d * 100000
             + e * 10000 + f * 1000 + g * 100 + h * 10 + i;
    }
    virtual int eleven(int a, int b, int c, int d, int e, int f, int g, int h,
                       int i, int j, int k) { return i * 100 + j * 10 + k; }
};
int main() {
    W w; W *p = &w;
    printf("%d %d\n", p->nine(1,2,3,4,5,6,7,8,9), p->eleven(1,2,3,4,5,6,7,8,9,1,2));
    return 0;
}
