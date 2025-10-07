#pragma once

#include "XYZ.h"

class Target {
public:
    Target();
    Target(float x, float y, float z);
    Target(const XYZ& pos);
    
    XYZ getCurrentPosition() const;
    void setPosition(const XYZ& pos);
    
private:
    XYZ position;
};
