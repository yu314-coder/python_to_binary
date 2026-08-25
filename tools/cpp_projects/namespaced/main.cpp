#include <stdio.h>
#include "util.hpp"
int main(){ std::string r = util::shout(std::string("hi")); printf("%s\n", r.c_str()); return 0; }
