#include <stdio.h>
#include <map>
int main(){ std::map<int,int> m; m[1]=10; m[2]=20; printf("%d %d %d\n",(int)m.size(),m[1],m[2]); return 0; }
