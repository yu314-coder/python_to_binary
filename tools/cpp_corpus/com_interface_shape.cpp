#include <stdio.h>
/* The same shape a generated COM header declares, written out so clang can
   read it too: an interface is a class of pure virtuals, and calling through
   a pointer to it goes via the table the object carries. */
typedef long HRESULT;
typedef struct _GUID { unsigned int a; unsigned short b; unsigned short c;
                       unsigned char d[8]; } GUID;
typedef const GUID *REFIID;
#define S_OK ((HRESULT)0)
#define FAILED(hr) ((HRESULT)(hr) < 0)

class IUnknown {
public:
  virtual HRESULT QueryInterface(REFIID riid, void **object) = 0;
  virtual unsigned long AddRef() = 0;
  virtual unsigned long Release() = 0;
  virtual ~IUnknown() { }
};
class ICounter : public IUnknown {
public: virtual HRESULT Next(int *out) = 0;
};
class Counter : public ICounter {
public:
  unsigned long refs; int value;
  Counter() { refs = 1; value = 0; }
  HRESULT QueryInterface(REFIID riid, void **object) { *object = this; AddRef(); return S_OK; }
  unsigned long AddRef() { refs = refs + 1; return refs; }
  unsigned long Release() { refs = refs - 1; return refs; }
  HRESULT Next(int *out) { value = value + 1; *out = value; return S_OK; }
};
static HRESULT twice(ICounter *counter, int *out) {
  HRESULT hr = counter->Next(out);
  if (FAILED(hr)) { return hr; }
  return counter->Next(out);
}
int main(){ Counter made; ICounter *counter = &made; int got = 0;
  HRESULT hr = twice(counter, &got);
  printf("%d %d %ld %lu\n", (int)(hr == S_OK), got, (long)hr, counter->AddRef());
  return 0; }
