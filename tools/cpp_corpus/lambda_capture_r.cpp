#include <stdio.h>
class Host { public: int sent; Host(){sent=0;} void send(int n){ sent += n; } };
int main(void){
    Host host; int total = 0;
    auto push = [&host, &total](int n) { host.send(n); total += n; };
    push(3); push(4);
    printf("%d %d\n", host.sent, total);
    return 0;
}
