/* py2bin's own <type_traits>, which is only possible because a class
   template may be written again for a shape of argument. */
#include <stdio.h>
#include <type_traits>

int main() {
    printf("%d %d %d %d %d\n",
           (int)std::is_same<int, int>::value,
           (int)std::is_same<int, char>::value,
           (int)std::is_pointer<char *>::value,
           (int)std::is_reference<int &>::value,
           (int)std::is_const<const int>::value);
    printf("%d %d %d %d\n",
           (int)std::is_integral<long>::value,
           (int)std::is_integral<double>::value,
           (int)std::is_floating_point<double>::value,
           (int)std::is_unsigned<unsigned int>::value);
    std::remove_reference<int &>::type a = 5;
    std::remove_pointer<char *>::type b = 'w';
    std::conditional<true, int, char>::type c = 6;
    std::conditional<false, int, char>::type d = 'y';
    printf("%d %c %d %c\n", a, b, c, d);
    return 0;
}
