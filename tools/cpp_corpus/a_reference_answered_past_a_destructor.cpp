// A function answering a reference, with an object to destroy on the way out: the answer is kept across the destructor.
#include <cstdio>
static int gone = 0;
struct Guard { ~Guard() { gone++; } };
struct Table { int cells[4]; int &at(int k) { Guard g; return cells[k]; } };
int main() { Table t; t.cells[2] = 5; t.at(2) += 1; printf("%d %d\n", t.cells[2], gone); return 0; }
