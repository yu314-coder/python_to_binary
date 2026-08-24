#include <stdio.h>
#include <stdexcept>
int risky(int n){ if (n < 0) throw std::runtime_error("neg"); return n; }
int wrap(int n){ return risky(n) * 2; }
int main(void){ try { wrap(-1); } catch (std::exception &e) { printf("%s\n", e.what()); }
  printf("%d\n", wrap(3)); return 0; }
