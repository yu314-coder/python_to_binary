#include <stdio.h>
#include <vector>
int main(){ std::vector<int> v; for(int i=0;i<5;i++) v.push_back(i);
  v.erase(v.begin()+1); printf("%d %d %d\n",(int)v.size(),v[0],v[1]); return 0; }
