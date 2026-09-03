#pragma once
template <class T> T clamp_to(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }
template <class T> struct Box { T held; T get() const { return held; } void put(T v) { held = clamp_to(v, (T)0, (T)9); } };
