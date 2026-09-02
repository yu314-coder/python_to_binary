// The project's own <vector> is the one this is compiled against, and
// py2bin's <bitset> - which is written on top of py2bin's own <string> -
// still works beside it. Overriding one of the headers py2bin ships must
// not take the rest of them apart.
#include <stdio.h>
#include <bitset>
#include <vector>
#include "tag.h"

int main(void) {
    Reading r(100);
    std::bitset<8> flags;
    flags.set(1);
    flags.set(3);
    printf("%d %d %s\n", STATION, r.fahrenheit(), flags.to_string().c_str());
    return 0;
}
