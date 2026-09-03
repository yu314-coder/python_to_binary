#pragma once
#include "vec.h"
struct Point { Vec at; Vec to(const Point &p) const { Vec d = { p.at.x - at.x, p.at.y - at.y }; return d; } };
