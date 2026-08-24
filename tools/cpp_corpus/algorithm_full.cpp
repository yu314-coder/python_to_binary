#include <stdio.h>
#include <algorithm>
#include <vector>
int main(void) {
    int raw[8]; raw[0]=9; raw[1]=2; raw[2]=7; raw[3]=2; raw[4]=5; raw[5]=1; raw[6]=8; raw[7]=3;
    std::sort(raw, raw + 8);
    for (int i = 0; i < 8; i++) printf("%d", raw[i]);
    printf("|");
    std::reverse(raw, raw + 8);
    for (int i = 0; i < 8; i++) printf("%d", raw[i]);
    printf("|%ld", std::count(raw, raw + 8, 2));
    printf("|%d", *std::max_element(raw, raw + 8));
    printf("|%d", *std::min_element(raw, raw + 8));
    int *at = std::find(raw, raw + 8, 5);
    printf("|%d", (int)(at - raw));
    std::vector<double> d;
    d.push_back(3.5); d.push_back(1.5); d.push_back(2.5);
    std::sort(d.begin(), d.end());
    printf("|%.1f%.1f%.1f", d[0], d[1], d[2]);
    int fill[4];
    std::fill(fill, fill + 4, 7);
    printf("|%d%d%d%d\n", fill[0], fill[1], fill[2], fill[3]);
    return 0;
}
