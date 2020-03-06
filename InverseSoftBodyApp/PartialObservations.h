#pragma once

#include <vector>
#include <Eigen/StdVector>
#include <cinder/params/Params.h>

#include <Commons.h>
#include <GridUtils.h>


namespace ar {

class PartialObservations
{
public:
    struct Camera
    {
        Vector2 position;
        Vector2 direction;
        Matrix2X observationPoints;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    };
    typedef std::vector<Camera, Eigen::aligned_allocator<Camera>> CameraVector;

private:
    const static uint32_t colors[20];

    CameraVector cameras_;
	GridUtils2D::grid_t referenceSdf_;
	GridUtils2D::grid_t currentSdf_;
	GridUtils2D::grid_t currentUx_;
	GridUtils2D::grid_t currentUy_;

    //AntTweakBarSettings
    int numCameras_;
    real fov_;
    int pointsPerCamera_;
    real noise_;

public:
    PartialObservations();
    ~PartialObservations();

    void initParams(cinder::params::InterfaceGlRef params);
    void draw();
    void setSdf(const GridUtils2D::grid_t& referenceSdf, const GridUtils2D::grid_t& currentSdf,
		const GridUtils2D::grid_t& currentUx, const GridUtils2D::grid_t& currentUy);
    Matrix2X getObservations() const;

private:
    void setNumCameras(int newCount);
    void recompute();
};

}