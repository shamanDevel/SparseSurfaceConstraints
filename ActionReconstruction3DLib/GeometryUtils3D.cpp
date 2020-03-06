#include "GeometryUtils3D.h"

#include <cinder/Log.h>

bool ar::geom3d::inside(const AABBBox & box, const Eigen::Vector3d & point)
{
    return (point.array() >= box.min.array()).all()
        && (point.array() <= box.max.array()).all();
}

std::vector<Eigen::Vector3d> ar::geom3d::intersect(const AABBBox & box, const Ray & r)
{
    double tmin, tmax, tymin, tymax, tzmin, tzmax;
    Eigen::Vector3d bounds[] = { box.min, box.max };

    tmin = (bounds[r.sign[0]].x() - r.origin.x()) * r.invdir.x();
    tmax = (bounds[1 - r.sign[0]].x() - r.origin.x()) * r.invdir.x();
    tymin = (bounds[r.sign[1]].y() - r.origin.y()) * r.invdir.y();
    tymax = (bounds[1 - r.sign[1]].y() - r.origin.y()) * r.invdir.y();

    if ((tmin > tymax) || (tymin > tmax))
        return std::vector<Eigen::Vector3d>(); //empty vector, no intersection
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[r.sign[2]].z() - r.origin.z()) * r.invdir.z();
    tzmax = (bounds[1 - r.sign[2]].z() - r.origin.z()) * r.invdir.z();

    if ((tmin > tzmax) || (tzmin > tmax))
        return std::vector<Eigen::Vector3d>(); //empty vector, no intersection
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    std::vector<Eigen::Vector3d> result;
    if (tmin >= 0) {
        result.push_back(r.origin + tmin * r.direction);
    }
    if (tmax >= 0) {
        result.push_back(r.origin + tmax * r.direction);
    }

    return result;
}

bool ar::geom3d::intersect(const AABBBox& box1, const AABBBox& box2)
{
    return (box1.max.array() >= box2.min.array()).all()
        && (box2.max.array() >= box1.min.array()).all();
}

ar::geom3d::AABBBox ar::geom3d::encloses(const std::vector<Eigen::Vector3d>& points)
{
    if (points.size() < 2) {
        CI_LOG_E("There has to be at least to points in the input list");
        return AABBBox();
    }
    
    Eigen::Array3d min = points[0];
    Eigen::Array3d max = points[0];
    for (int i = 1; i < points.size(); ++i) {
        min = min.min(points[i].array());
        max = max.max(points[i].array());
    }
    return AABBBox(min.matrix(), max.matrix());
}

ar::geom3d::AABBBox ar::geom3d::intersect(const AABBBox & box, const Pyramid & pyramid)
{
    std::vector<Eigen::Vector3d> controlPoints;
    //1. Intersect pyramid rays with with box
    for (const Eigen::Vector3d& dir : pyramid.edgeRays)
    {
        Ray ray(pyramid.center, dir);
        std::vector<Eigen::Vector3d> intersections = intersect(box, ray);
        controlPoints.insert(controlPoints.end(), intersections.begin(), intersections.end());
    }
    //2. if the pyramid center is inside the box, add it also to the list
    if (inside(box, pyramid.center)) {
        controlPoints.push_back(pyramid.center);
    }
    //3. return enclosing box
    return encloses(controlPoints);
}

static inline Eigen::Vector3d toEigen(const glm::vec3& vec)
{
    return Eigen::Vector3d(vec.x, vec.y, vec.z);
}

ar::geom3d::Pyramid ar::geom3d::pyramidFromCamera(const ar3d::DataCamera & cam)
{
    Pyramid p;
    p.center = toEigen(cam.location);
    p.edgeRays.push_back(toEigen(cam.getWorldCoordinates(glm::vec3(0, 0, 0)) - cam.location).normalized());
    p.edgeRays.push_back(toEigen(cam.getWorldCoordinates(glm::vec3(1, 0, 0)) - cam.location).normalized());
    p.edgeRays.push_back(toEigen(cam.getWorldCoordinates(glm::vec3(1, 1, 0)) - cam.location).normalized());
    p.edgeRays.push_back(toEigen(cam.getWorldCoordinates(glm::vec3(0, 1, 0)) - cam.location).normalized());
    return p;
}

ar::geom3d::Ray::Ray(const Eigen::Vector3d & origin, const Eigen::Vector3d & normalizedDirection)
    : origin(origin)
    , direction(normalizedDirection)
{
    invdir = 1.0 / direction.array();
    sign[0] = (invdir.x() < 0);
    sign[1] = (invdir.y() < 0);
    sign[2] = (invdir.z() < 0);
}
