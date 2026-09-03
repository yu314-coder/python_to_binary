#include <stdio.h>
#include "algo.hpp"
int main(void) { Box<int> bi; bi.put(42); Box<double> bd; bd.put(-2.5); printf("%d %g %d\n", bi.get(), bd.get(), clamp_to(7, 1, 5)); return 0; }
