#include <stdio.h>
#include <fstream>
int main(void) { std::filesystem::path p("dir/leaf.txt"); printf("%s\n", mini::name_of(p).c_str()); return 0; }
