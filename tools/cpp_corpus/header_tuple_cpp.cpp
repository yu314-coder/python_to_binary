/* <tuple>: made both ways and read by index. */
#include <tuple>
#include <cstdio>

int main() {
    std::tuple<int, double, char> three(1, 2.5, 'z');
    printf("%d %.1f %c\n", std::get<0>(three), std::get<1>(three),
           std::get<2>(three));
    std::tuple<int, int> two = std::make_tuple(7, 8);
    printf("%d %d\n", std::get<0>(two), std::get<1>(two));
    return 0;
}
