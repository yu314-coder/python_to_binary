#include <stdio.h>
class Log { public: int n; Log(){n=0;} void put(int v){ n += v; } };
int main(void){
    Log log; int scale = 10; int copied = 5;
    auto go = [&log, copied](int v) { log.put(v * copied); };
    go(2);
    auto plain = [scale](int v) { return v * scale; };
    printf("%d %d\n", log.n, plain(3));
    return 0;
}
