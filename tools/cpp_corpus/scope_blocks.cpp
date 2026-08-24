#include <stdio.h>
class R { public: int id; R() { id = 1; } ~R() { printf("gone\n"); } };
int main(void) { printf("before\n"); { R r; printf("inside %d\n", r.id); } printf("after\n"); return 0; }
