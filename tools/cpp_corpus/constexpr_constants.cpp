#include <cstdio>
constexpr int kLimit = 12;
constexpr char kName[] = "sidecar";
constexpr double kRatio = 1.5;
constexpr int twice(int n) { return n * 2; }
class Box {
public:
    static constexpr int kSlots = 4;
    int used = 0;
};
int main() {
    int room[kLimit];
    room[0] = twice(kLimit);
    printf("%d %s %.1f %d %d\n", room[0], kName, kRatio, Box::kSlots, kLimit);
    return 0;
}
