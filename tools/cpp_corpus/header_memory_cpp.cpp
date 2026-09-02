/* <memory> as py2bin has it: `unique_ptr` owns and frees, `shared_ptr` holds
   and hands over. Neither `make_unique` nor `use_count` is here - see the
   note in the header, which says what this subset is. */
#include <memory>
#include <cstdio>

struct Thing {
    int held;
    Thing(int value) : held(value) {}
};

int main() {
    std::unique_ptr<Thing> owned(new Thing(4));
    printf("%d %d\n", owned->held, (*owned).held);
    Thing *let_go = owned.release();
    printf("%d\n", owned.get() == 0);
    delete let_go;

    std::shared_ptr<Thing> first(new Thing(7));
    std::shared_ptr<Thing> second = first;
    printf("%d %d\n", second->held, first.get() == second.get());
    return 0;
}
