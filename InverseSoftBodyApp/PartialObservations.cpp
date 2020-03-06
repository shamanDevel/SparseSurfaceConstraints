#include "PartialObservations.h"

#include <cinder/gl/gl.h>
#include <random>

// 0: raytracing the result SDF
// 1: intersection with boundary lines
#define ALGORITHM 1

#if ALGORITHM==1
#include <GridVisualization.h>
#include <GeometryUtils2D.h>
#endif

namespace ar {

const uint32_t PartialObservations::colors[] = {
    0xe6194b,
    0x3cb44b,
    0xffe119,
    0x0082c8,
    0xf58231,
    0x911eb4,
    0x46f0f0,
    0xf032e6,
    0xd2f53c,
    0xfabebe,
    0x008080,
    0xe6beff,
    0xaa6e28,
    0xfffac8,
    0x800000,
    0xaaffc3,
    0x808000,
    0xffd8b1,
    0x000080,
    0x808080
};

PartialObservations::PartialObservations()
    : numCameras_(0)
    , fov_(30)
    , pointsPerCamera_(20)
    , noise_(0.01)
{
    setNumCameras(5);
}


PartialObservations::~PartialObservations()
{
}

void PartialObservations::initParams(cinder::params::InterfaceGlRef params)
{
    params->addParam("PartialObservations-NumCameras", &numCameras_)
        .label("Num Cameras").group("Partial Observations").min(1).max(20)
        .accessors([this](int value){setNumCameras(value);}, [this](){return numCameras_;});
    params->addParam("PartialObservations-FOV", &fov_)
        .label("FOV").group("Partial Observations").min(1).step(1).max(60)
        .accessors([this](real value){fov_ = value; recompute();}, [this](){return fov_;});
    params->addParam("PartialObservations-PointsPerCamera", &pointsPerCamera_)
        .label("Points per Camera").group("Partial Observations").min(1)
        .accessors([this](int value){pointsPerCamera_ = value; recompute();}, [this](){return pointsPerCamera_;});
    params->addParam("PartialObservations-Noise", &noise_)
        .label("Noise").group("Partial Observations").min(0.001).step(0.001)
        .accessors([this](real value){noise_ = value; recompute();}, [this](){return noise_;});
}

void PartialObservations::draw()
{
    using namespace cinder;
    Vector2 offset(0,0);
    if (referenceSdf_.size() > 0) offset = Vector2(0.5 / referenceSdf_.rows(), 0.5 / referenceSdf_.cols());
    for (size_t i=0; i<cameras_.size(); ++i)
    {
        const Camera& cam = cameras_[i];
        vec2 pos = vec2(cam.position.x()+offset.x(), cam.position.y()+offset.y()) - vec2(0.5, 0.5);
        vec2 dir = vec2(cam.direction.x(), cam.direction.y());
        pos.y = -pos.y; dir.y = -dir.y;

        gl::color(Color8u::hex(colors[i]));
        //draw outline of the camera
        gl::drawSolidCircle(pos, 0.005, 4);
        vec2 dir1 = vec2(
            dir.x*cos(M_PI*fov_/180) - dir.y*sin(M_PI*fov_/180),
            dir.y*cos(M_PI*fov_/180) + dir.x*sin(M_PI*fov_/180));
        vec2 dir2 = vec2(
            dir.x*cos(-M_PI*fov_/180) - dir.y*sin(-M_PI*fov_/180),
            dir.y*cos(-M_PI*fov_/180) + dir.x*sin(-M_PI*fov_/180));
        gl::drawLine(pos, pos + dir1 * 0.1f);
        gl::drawLine(pos, pos + dir2 * 0.1f);
        gl::drawLine(pos + dir1 * 0.1f, pos + dir2 * 0.1f);
        //draw points
        for (int j=0; j<cam.observationPoints.cols(); ++j)
        {
            Vector2 p = cam.observationPoints.col(j) + offset - Vector2(0.5, 0.5);
            p.y() = -p.y();
            gl::drawSolidCircle(vec2(p.x(), p.y()), 0.006, 8);
        }
    }
}

void PartialObservations::setSdf(const GridUtils2D::grid_t& referenceSdf, const GridUtils2D::grid_t& currentSdf,
	const GridUtils2D::grid_t& currentUx, const GridUtils2D::grid_t& currentUy)
{
	referenceSdf_ = referenceSdf;
	currentSdf_ = currentSdf;
	currentUx_ = currentUx;
	currentUy_ = currentUy;
	recompute();
}

Matrix2X PartialObservations::getObservations() const
{
    Eigen::Index size = 0;
    for (const auto& cam : cameras_)
        size += cam.observationPoints.cols();
    if (size == 0) return Matrix2X();
    Matrix2X points; points.resize(2, size);
    Eigen::Index i = 0;
    for (const auto& cam : cameras_)
    {
        points.block(0, i, 2, cam.observationPoints.cols()) = cam.observationPoints;
        i += cam.observationPoints.cols();
    }
    return points;
}

void PartialObservations::setNumCameras(int newCount)
{
    cameras_.resize(newCount);
    numCameras_ = newCount;

    //distribute around center
    Vector2 center(0.5, 0.5);
    real radius = 0.5;
    for (int i=0; i<newCount; ++i)
    {
        real angle = M_PI * 2 * i / newCount + 0.1;
        Vector2 position = center + radius * Vector2(sin(angle), -cos(angle));
        Vector2 direction = (center - position).normalized();
        cameras_[i].position = position;
        cameras_[i].direction = direction;
        cameras_[i].observationPoints = Matrix2X();
    }

    recompute();
}

void PartialObservations::recompute()
{
    if (referenceSdf_.size() == 0) return;

    static std::default_random_engine rnd;
    std::normal_distribution<real> distr(0, noise_ * 0.1 / referenceSdf_.rows());

#if ALGORITHM==1
	const auto lines = GridVisualization::getGridBoundaryLines(referenceSdf_, currentUx_, currentUy_);
#endif

    for (auto& cam : cameras_)
    {
        Vector2 dir1 = Vector2(
            cam.direction.x()*cos(M_PI*fov_/180) - cam.direction.y()*sin(M_PI*fov_/180),
            cam.direction.y()*cos(M_PI*fov_/180) + cam.direction.x()*sin(M_PI*fov_/180));
        Vector2 dir2 = Vector2(
            cam.direction.x()*cos(-M_PI*fov_/180) - cam.direction.y()*sin(-M_PI*fov_/180),
            cam.direction.y()*cos(-M_PI*fov_/180) + cam.direction.x()*sin(-M_PI*fov_/180));
        std::vector<Vector2, Eigen::aligned_allocator<Vector2>> points;
        for (int j=0; j<pointsPerCamera_; ++j)
        {
            Vector2 start = cam.position;
            Vector2 dir = utils::mix(j / (pointsPerCamera_-1.0), dir1, dir2).normalized();

			bool found = false;
			Vector2 pos(0, 0);
#if ALGORITHM==0
            //Algorithm 1: raytracing of the result SDF
            real stepsize = 0.1 / currentSdf_.rows();
            for (int k=0; ; ++k)
            {
                pos = start + k * stepsize * dir;
                if (pos.x() <= 0 || pos.x()>=1 || pos.y()<=0 || pos.y()>=1) break; //out of bounds
                real value = GridUtils2D::getClampedF(currentSdf_, pos.x()*currentSdf_.rows(), pos.y()*currentSdf_.cols());
                if (utils::insideEq(value))
                {
                    found = true;
                    break;
                }
            }
#else
			//Algorithm 2: intersesction with the boundary lines
			real dist = 1e+20;
			geom2d::line ray = geom2d::lineByDirection({ start.x(), start.y() }, { dir.x(), dir.y() });
			for (const auto& line : lines)
			{
				geom2d::line l = geom2d::lineByPoints({ line.first.x(), line.first.y() }, { line.second.x(), line.second.y() });
				geom2d::point p = geom2d::intersection(ray, l);
				if (!geom2d::between({ line.first.x(), line.first.y() }, { line.second.x(), line.second.y() }, p)) continue; //not in the line segment
				Vector2 p2(p.first, p.second);
				real d = (p2 - start).norm();
				if (d < dist)
				{
					dist = d;
					found = true;
					pos = p2;
				}
			}

#endif
            if (found)
            {
                //add some noise and add it to the points
                pos += Vector2(distr(rnd), distr(rnd));
                points.push_back(pos);
            }
        }
        //write into observation matrix
        if (points.empty())
            cam.observationPoints = Matrix2X();
        else {
            cam.observationPoints.resize(2, points.size());
            for (int k=0; k<points.size(); ++k) cam.observationPoints.col(k) = points[k];
        }
    }
}

}
