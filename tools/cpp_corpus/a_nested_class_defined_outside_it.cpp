// A class declared inside another and defined below it - `struct Session;`
// in the header, `struct Owner::Session { ... };` in the source - which is how
// a class keeps a type private and its shape out of the header. The lifting
// moves a class written *inside* another and never saw this one, so nothing
// declared the type at all: the C held a `shared_ptr` pointing at a struct
// that was nowhere. The qualifier comes off the definition and off every
// mention, and the declaration left inside the class goes with it. The holder
// beside it spells one template argument and leaves the rest to the call.
#include <cstdio>
#include <memory>
#include <string>
class Host {
public:
    struct Room;
    Host() { count_ = 0; }
    int count_;
    std::shared_ptr<Room> open(int id);
    int size_of(std::shared_ptr<Room> room);
};
struct Host::Room {
    int id;
    std::string name;
    Room(int given) { id = given; name = "room"; }
};
std::shared_ptr<Host::Room> Host::open(int id) {
    count_ = count_ + 1;
    return std::make_shared<Room>(id);
}
int Host::size_of(std::shared_ptr<Room> room) { Room *held = room.get(); return held->id + (int)held->name.size(); }
int main() {
    Host h;
    std::shared_ptr<Host::Room> r = h.open(5);
    Host::Room *held = r.get();
    printf("%d %d %s\n", h.size_of(r), h.count_, held->name.c_str());
    return 0;
}
