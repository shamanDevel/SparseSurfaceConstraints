#include "WorldGrid.h"

ar3d::WorldGrid::WorldGrid(int voxelResolution, const Eigen::Vector3i & offset, const Eigen::Vector3i & size)
    : voxelResolution_(voxelResolution)
    , offset_(offset)
    , size_(size)
{
    if (voxelResolution < 1) {
        throw std::exception("voxel resolution must be a positive integer");
    }
}
