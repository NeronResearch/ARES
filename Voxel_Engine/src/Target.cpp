#include "Target.h"

// Target Implementation
Target::Target() : position(0,0,0), visible(false) {}
Target::Target(float x, float y, float z) : position(x, y, z), visible(false) {}
Target::Target(const XYZ& pos) : position(pos), visible(false) {}
Target::Target(const XYZ& pos, bool visible) : position(pos), visible(visible) {}
XYZ Target::getCurrentPosition() const { return position; }
bool Target::isVisible() const { return visible; }
void Target::setPosition(const XYZ& pos) { position = pos; }
