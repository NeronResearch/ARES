#include "Target.h"

// Target Implementation
Target::Target() : position(0,0,0) {}
Target::Target(float x, float y, float z) : position(x, y, z) {}
Target::Target(const XYZ& pos) : position(pos) {}
XYZ Target::getCurrentPosition() const { return position; }
void Target::setPosition(const XYZ& pos) { position = pos; }
