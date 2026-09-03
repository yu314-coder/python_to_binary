#pragma once
struct Vec { int x, y; int dot(const Vec &o) const { return x * o.x + y * o.y; } };
