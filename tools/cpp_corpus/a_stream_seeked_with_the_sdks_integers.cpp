/* `memoryStream->Seek(zero, STREAM_SEEK_SET, nullptr)` and
   `stat.cbSize.QuadPart` - IStream spoken the way the SDK spells it, against
   the header py2bin ships.

   That header writes the eight-byte integers the SDK passes by value as the
   integers they are: a struct passed by value through a foreign table is the
   one thing py2bin cannot spell, and on Windows an eight-byte struct travels
   where an eight-byte integer does. A program written against the SDK hands
   the struct, and a pointer to one - so `Seek` reached the C stage with a
   `LARGE_INTEGER` where a `long long` was wanted, and `stat.cbSize.QuadPart`
   read a member out of a member declared as a plain integer. Both are
   refused lines of correct Windows C++.

   The stream is one of the program's own, behind the interface: every value
   the call hands over is kept and printed, so a value that arrived wrong, or
   a slot that dispatched to the wrong method, is a wrong number. There is no
   clang++ reference - the header is py2bin's - and the numbers are what the
   calls below spell: 77 0 1077 12 4097. */
#include <unknwn.h>
#include <objidl.h>
#include <cstdio>

class Remembered : public IStream {
public:
    long long moved;
    unsigned long origin;
    unsigned long long size;
    Remembered() : moved(-1), origin(99), size(0) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **object) {
        *object = this;
        return S_OK;
    }
    unsigned long STDMETHODCALLTYPE AddRef() { return 2; }
    unsigned long STDMETHODCALLTYPE Release() { return 1; }
    HRESULT STDMETHODCALLTYPE Read(void *pv, unsigned long cb, unsigned long *read) {
        if (read) *read = cb;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE Write(const void *pv, unsigned long cb,
                                    unsigned long *written) {
        if (written) *written = cb;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE Seek(long long move, unsigned long how,
                                   unsigned long long *position) {
        moved = move;
        origin = how;
        if (position) *position = (unsigned long long)(move + 1000);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetSize(unsigned long long wanted) {
        size = wanted;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CopyTo(IStream *other, unsigned long long cb,
                                     unsigned long long *read,
                                     unsigned long long *written) {
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE Commit(unsigned long flags) { return S_OK; }
    HRESULT STDMETHODCALLTYPE Revert() { return S_OK; }
    HRESULT STDMETHODCALLTYPE LockRegion(unsigned long long offset,
                                         unsigned long long cb, unsigned long type) {
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE UnlockRegion(unsigned long long offset,
                                           unsigned long long cb, unsigned long type) {
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE Stat(STATSTG *out, unsigned long flag) {
        out->cbSize.QuadPart = 4096 + flag;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE Clone(IStream **out) {
        /* Not `*out = this`: a pointer to this class assigned through a
           pointer to its base is a conversion the translator does not write
           yet, and the C stage refuses it. Nothing here needs it. */
        *out = 0;
        return S_OK;
    }
};

int main() {
    Remembered remembered;
    IStream *stream = &remembered;
    LARGE_INTEGER zero{};
    zero.QuadPart = 77;
    ULARGE_INTEGER landed{};
    ULARGE_INTEGER grown{};
    grown.QuadPart = 12;
    stream->Seek(zero, STREAM_SEEK_SET, &landed);
    stream->SetSize(grown);
    STATSTG stat{};
    stream->Stat(&stat, 1);
    printf("%lld %lu %llu %llu %llu\n", remembered.moved, remembered.origin,
           landed.QuadPart, remembered.size, stat.cbSize.QuadPart);
    return 0;
}
