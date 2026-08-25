#include <stdio.h>
int main(void){
  printf("[%10.3e][%-10.3e][%010.3f]\n", -1234.5, -1234.5, -1234.5);
  printf("[%6.1f][%+8.2f][%2.4f]\n", 0.0, 2.5, 1.0);
  printf("[%3d|%-3d|%d]\n", 12345, 12345, 0);
  return 0; }
