#include <cstdio>
#include <tuple>
int main() {
    std::tuple<int, int> a(1, 2);
    std::tuple<int, char, double> b(7, 'z', 1.5);
    printf("%d %d %d %c %.1f\n", std::get<0>(a), std::get<1>(a),
           std::get<0>(b), std::get<1>(b), std::get<2>(b));
    return 0;
}
