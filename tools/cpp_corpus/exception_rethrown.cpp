#include <stdio.h>
#include <stdexcept>
static int risky(int n){ if (n < 0) throw std::runtime_error("neg"); return n; }
int main(){
  try { try { risky(-1); } catch (std::exception &e) { throw; } }
  catch (std::exception &e) { printf("caught %s\n", e.what()); }
  return 0; }
