#include <stdio.h>
#include <sstream>
#include <string>
int main(){ std::ostringstream o; o << "a=" << 1 << ",b=" << 'x' << "," << std::string("s");
  printf("%s\n", o.str().c_str()); return 0; }
