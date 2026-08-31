#include <unknwn.h>
#include <oaidl.h>
#include <cstdio>

MIDL_INTERFACE("2933BF80-7B36-11d2-B20E-00C04F983E60")
IXMLDOMNode : public IDispatch
{
public:
    virtual HRESULT STDMETHODCALLTYPE get_nodeName(int *name) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_nodeValue(int *value) = 0;
};

class Node : public IXMLDOMNode {
public:
    long refs;
    Node() : refs(1) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **out) { *out = 0; return 1; }
    unsigned long STDMETHODCALLTYPE AddRef() { return ++refs; }
    unsigned long STDMETHODCALLTYPE Release() { return --refs; }
    HRESULT STDMETHODCALLTYPE GetTypeInfoCount(unsigned int *n) { *n = 11; return 2; }
    HRESULT STDMETHODCALLTYPE GetTypeInfo(unsigned int i, LCID l, ITypeInfo **t) { return 3; }
    HRESULT STDMETHODCALLTYPE GetIDsOfNames(REFIID r, LPOLESTR *n, unsigned int c, LCID l, DISPID *d) { return 4; }
    HRESULT STDMETHODCALLTYPE Invoke(DISPID d, REFIID r, LCID l, unsigned short f, DISPPARAMS *p, VARIANT *v, EXCEPINFO *e, unsigned int *a) { return 5; }
    HRESULT STDMETHODCALLTYPE get_nodeName(int *name) { *name = 77; return 6; }
    HRESULT STDMETHODCALLTYPE get_nodeValue(int *value) { *value = 88; return 7; }
};

int main() {
    Node n;
    IXMLDOMNode *p = &n;
    IDispatch *d = &n;
    IUnknown *u = &n;
    int name = 0, value = 0;
    unsigned int count = 0;
    int a = (int)p->get_nodeName(&name);
    int b = (int)p->get_nodeValue(&value);
    int c = (int)d->GetTypeInfoCount(&count);
    int e = (int)d->Invoke(0, 0, 0, 0, 0, 0, 0, 0);
    printf("%d %d %d %d %d %d %d %d\n", a, name, b, value, c, (int)count, e, (int)u->AddRef());
    return 0;
}
