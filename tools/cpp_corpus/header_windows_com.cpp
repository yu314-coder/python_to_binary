/* The COM headers py2bin ships, which are C++ rather than C: <unknwn.h> for
   IUnknown, <objidl.h> and <oaidl.h> for the interfaces a generated header
   names in its signatures, and <wrl.h> for the pointer that counts. None of
   them is guarded: COM's shapes are the same on every machine and py2bin
   writes them out itself, so a program that declares an interface builds for
   all six - which is the whole point of shipping them rather than reading
   the SDK's. */
#include <unknwn.h>
#include <objidl.h>
#include <oaidl.h>
#include <wrl.h>
#include <cstdio>

class Thing : public IUnknown {
public:
    unsigned long refs;
    Thing() { refs = 1; }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **out) {
        *out = (void *)this;
        return S_OK;
    }
    unsigned long STDMETHODCALLTYPE AddRef() { refs = refs + 1; return refs; }
    unsigned long STDMETHODCALLTYPE Release() { refs = refs - 1; return refs; }
};

int main() {
    Thing thing;
    HRESULT answered = S_OK;
    IStream *stream = 0;
    IDispatch *dispatch = 0;
    /* Named rather than written as `&thing`: ComPtr takes either a `T *` or
       another ComPtr, and the address of a derived object left py2bin unable
       to say which of the two was meant. */
    IUnknown *raw = &thing;
    Microsoft::WRL::ComPtr<IUnknown> held(raw);
    /* S_OK and S_FALSE only. Whether E_POINTER is a failure depends on how
       wide a `long` is, and that is four bytes on Windows and eight
       everywhere else - so it is not the same answer on all six machines and
       has no business being printed. */
    printf("%d %d\n", SUCCEEDED(answered), SUCCEEDED(S_FALSE));
    printf("%lu\n", thing.refs);
    printf("%d %d\n", stream == 0, dispatch == 0);
    printf("%d\n", held.Get() == raw);
    return 0;
}
