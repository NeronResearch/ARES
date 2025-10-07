#pragma once
#include "XYZ.h"
#include "Voxel.h"
#include <vector>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

class VoxelGrid {
public:
    VoxelGrid() : size(0, 0, 0), origin(0, 0, 0), sx(0), sy(0), sz(0) {}

    VoxelGrid(const XYZ& size, const XYZ& origin, float voxelSize = 1.0f)
        : origin(origin), voxelSize(voxelSize) {
        sx = static_cast<int>(std::ceil(size.getX() / voxelSize));
        sy = static_cast<int>(std::ceil(size.getY() / voxelSize));
        sz = static_cast<int>(std::ceil(size.getZ() / voxelSize));

        this->size = XYZ(sx * voxelSize, sy * voxelSize, sz * voxelSize);
        const size_t total = static_cast<size_t>(sx) * sy * sz;
        voxels.resize(total);

#ifdef _OPENMP
        #pragma omp parallel for collapse(3)
#endif
        for (int z = 0; z < sz; ++z) {
            for (int y = 0; y < sy; ++y) {
                for (int x = 0; x < sx; ++x) {
                    voxels[indexFromIndices(x, y, z)].setPosition(XYZ(
                        origin.getX() + x * voxelSize + voxelSize / 2.0f,
                        origin.getY() + y * voxelSize + voxelSize / 2.0f,
                        origin.getZ() + z * voxelSize + voxelSize / 2.0f
                    ));
                }
            }
        }
    }

    inline Voxel& at(const XYZ& worldCoords) {
        auto [xi, yi, zi] = worldToIndices(worldCoords);
        return voxels[indexFromIndices(xi, yi, zi)];
    }

    inline const Voxel& at(const XYZ& worldCoords) const {
        auto [xi, yi, zi] = worldToIndices(worldCoords);
        return voxels[indexFromIndices(xi, yi, zi)];
    }

    inline size_t indexFromIndices(int xi, int yi, int zi) const {
        return (static_cast<size_t>(zi) * sy + yi) * sx + xi;
    }

    inline std::tuple<int, int, int> worldToIndices(const XYZ& worldCoords) const {
        float voxelSize = this->voxelSize;
        int xi = static_cast<int>(std::floor((worldCoords.getX() - origin.getX()) / voxelSize));
        int yi = static_cast<int>(std::floor((worldCoords.getY() - origin.getY()) / voxelSize));
        int zi = static_cast<int>(std::floor((worldCoords.getZ() - origin.getZ()) / voxelSize));

        if (xi < 0 || xi >= sx || yi < 0 || yi >= sy || zi < 0 || zi >= sz) {
            throw std::out_of_range("VoxelGrid::worldToIndices() - coordinates out of bounds");
        }
        return {xi, yi, zi};
    }

    void setOrigin(const XYZ& o) { origin = o; }
    XYZ getOrigin() const { return origin; }
    XYZ getSize() const { return size; }
    int getSizeX() const { return sx; }
    int getSizeY() const { return sy; }
    int getSizeZ() const { return sz; }
    float getVoxelSize() const { return voxelSize; }
    
    std::vector<Voxel> voxels;

private:
    XYZ size;
    XYZ origin;
    int sx, sy, sz; 
    float voxelSize;
};