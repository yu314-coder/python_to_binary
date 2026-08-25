#include <stdio.h>
class Widget { public: int deleted; int deleteLater; Widget() { deleted = 0; deleteLater = 3; }
  int deleteAll() { deleted = 1; return deleteLater; } };
int main() { Widget w; printf("%d %d\n", w.deleteAll(), w.deleted); return 0; }
