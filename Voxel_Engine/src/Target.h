#pragma once

#include "XYZ.h"

class Target {
public:
    Target();
    Target(float x, float y, float z);
    Target(const XYZ& pos);
    Target(const XYZ& pos, bool visible);
    
    XYZ getCurrentPosition() const;
    bool isVisible() const;
    void setPosition(const XYZ& pos);
    
private:
    XYZ position;
    bool visible;
};
