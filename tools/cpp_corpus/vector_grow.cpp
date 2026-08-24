#include <stdio.h>
#include <vector>
int main(void){ std::vector<int> v; long t = 0;
  for (int i = 0; i < 500; i++) v.push_back(i);
  for (unsigned long i = 0; i < v.size(); i++) t += v[i];
  printf("%lu %ld %d\n", (unsigned long)v.size(), t, v[499]); return 0; }
