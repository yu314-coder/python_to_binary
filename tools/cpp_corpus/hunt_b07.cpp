#include <stdio.h>
class Peer;
class Owner { friend class Peer; public: int secret; Owner() { secret = 42; } };
class Peer { public: int read(Owner &o) { return o.secret; } };
int main() { Owner o; Peer p; printf("%d\n", p.read(o)); return 0; }
