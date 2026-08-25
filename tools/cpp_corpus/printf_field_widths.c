#include <stdio.h>
int main(void){
  printf("[%5d][%-5d][%05d][%+d][% d]\n", 42, 42, 42, 42, 42);
  printf("[%5d][%-5d][%05d][%+d]\n", -42, -42, -42, -42);
  printf("[%8s][%-8s][%3c]\n", "hi", "hi", 'x');
  printf("[%08x][%4X]\n", 1065353216u, 255u);
  return 0; }
