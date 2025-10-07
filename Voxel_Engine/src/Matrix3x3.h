#pragma once
#include "XYZ.h"

class Matrix3x3 {
public:
    float m[3][3];

    Matrix3x3() {
        m[0][0] = 1; m[0][1] = 0; m[0][2] = 0;
        m[1][0] = 0; m[1][1] = 1; m[1][2] = 0;
        m[2][0] = 0; m[2][1] = 0; m[2][2] = 1;
    }

    XYZ operator*(const XYZ& v) const {
        return {
            m[0][0]*v.getX() + m[0][1]*v.getY() + m[0][2]*v.getZ(),
            m[1][0]*v.getX() + m[1][1]*v.getY() + m[1][2]*v.getZ(),
            m[2][0]*v.getX() + m[2][1]*v.getY() + m[2][2]*v.getZ()
        };
    }
};