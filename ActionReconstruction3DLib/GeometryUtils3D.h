#pragma once

#include <Eigen/Core>
#include <vector>

#include "DataCamera.h"

namespace ar {
namespace geom3d {

    struct AABBBox
    {
        Eigen::Vector3d min;
        Eigen::Vector3d max;
        AABBBox() : min(0, 0, 0), max(0, 0, 0) {}
        AABBBox(const Eigen::Vector3d& min, const Eigen::Vector3d& max) : min(min), max(max) {}
        bool isValid() const { return (min.array() < max.array()).all(); }
    };

    struct Pyramid
    {
        Eigen::Vector3d center;
        std::vector<Eigen::Vector3d> edgeRays;
        Pyramid() : center(), edgeRays() {}
        Pyramid(const Eigen::Vector3d& center, const std::vector<Eigen::Vector3d>& edgeRays)
            : center(center), edgeRays(edgeRays) {}
    };

    struct Ray
    {
        Eigen::Vector3d origin;
        Eigen::Vector3d direction;
        Eigen::Vector3d invdir;
        int sign[3];

        Ray() : origin(), direction() {}
        Ray(const Eigen::Vector3d& origin, const Eigen::Vector3d& normalizedDirection);
    };

    //Tests if the given point is inside (or touching) the box
    bool inside(const AABBBox& box, const Eigen::Vector3d& point);

    //Returns all intersection points (zero, one, two) of the box and the ray
    std::vector<Eigen::Vector3d> intersect(const AABBBox& box, const Ray& ray);

    //Checks if two AABB-boxes intersect
    bool intersect(const AABBBox& box1, const AABBBox& box2);

    //Returns the box that encloses the given list of points
    AABBBox encloses(const std::vector<Eigen::Vector3d>& points);

    //Returns the box that encloses the intersection of the given box and pyramid
    AABBBox intersect(const AABBBox& box, const Pyramid& pyramid);

    //Constructs a pyramid from the camera frustrum
    Pyramid pyramidFromCamera(const ar3d::DataCamera& cam);
}
}