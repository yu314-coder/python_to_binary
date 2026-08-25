#include <stdio.h>
#include <memory>
int main(){ std::unique_ptr<int> p(new int(5)); printf("%d\n", *p); return 0; }
