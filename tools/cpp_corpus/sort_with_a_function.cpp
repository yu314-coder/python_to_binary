#include <stdio.h>
#include <algorithm>
#include <vector>
static bool bigger(int a,int b){ return a>b; }
int main(){ std::vector<int> v; v.push_back(1); v.push_back(9); v.push_back(5);
  std::sort(v.begin(), v.end(), bigger); printf("%d%d%d\n",v[0],v[1],v[2]); return 0; }
