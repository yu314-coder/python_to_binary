#include <stdio.h>
#include <stdexcept>
#include <vector>
static int at(const std::vector<int> &v, int i){
  if (i < 0 || i >= (int)v.size()) throw std::out_of_range("bad index");
  return v[i]; }
int main(){ std::vector<int> v; v.push_back(5);
  printf("%d ", at(v, 0));
  try { at(v, 9); } catch (std::exception &e) { printf("%s\n", e.what()); }
  return 0; }
