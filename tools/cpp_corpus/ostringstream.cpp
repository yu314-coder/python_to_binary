#include <stdio.h>
#include <sstream>
#include <string>
int main(){ std::ostringstream o; o << "n=" << 42; printf("%s\n", o.str().c_str()); return 0; }
