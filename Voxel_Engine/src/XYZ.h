#pragma once
#include <cmath>

struct XYZ {
    float x, y, z;
    
    XYZ(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    
    float getX() const { return x; }
    float getY() const { return y; }
    float getZ() const { return z; }
    
    void setX(float val) { x = val; }
    void setY(float val) { y = val; }
    void setZ(float val) { z = val; }
    
    // Vector operations
    XYZ operator-(const XYZ& other) const {
        return XYZ(x - other.x, y - other.y, z - other.z);
    }
    
    XYZ operator+(const XYZ& other) const {
        return XYZ(x + other.x, y + other.y, z + other.z);
    }
    
    XYZ operator*(float scalar) const {
        return XYZ(x * scalar, y * scalar, z * scalar);
    }
    
    float magnitude() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    XYZ normalize() const {
        float mag = magnitude();
        if (mag > 0) {
            return XYZ(x/mag, y/mag, z/mag);
        }
        return XYZ(0, 0, 0);
    }
};

struct XY {
    float x;
    float y;
};
