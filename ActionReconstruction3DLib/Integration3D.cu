#include "Integration3D.h"

#include <random>
#include <bitset>

#include "HaltonSequence.h"
#include "MarchingCubes.h"
#include "Utils3D.h"
#include "GeometryUtils3D.h"

namespace ar3d {
	Integration3D::InterpolationWeight_t Integration3D::volumeIntegral(const InputPhi_t& phi, real h)
	{
		return volumeIntegralLinear(phi, h);
	}

	Integration3D::InterpolationWeight_t Integration3D::surfaceIntegral(const InputPhi_t& phi, real h)
	{
		return surfaceIntegralMC(phi, h);
	}

	real Integration3D::interpolate(const InputPhi_t& phi, real x, real y, real z)
	{
		real4 v1 = (1 - z) * phi.first + z * phi.second;
		return (1 - x)*(1 - y) * v1.x + x * (1 - y)*v1.y + (1 - x)*y*v1.z + x * y*v1.w;
	}

	// volume is sampled
	Integration3D::InterpolationWeight_t Integration3D::volumeIntegralSampled(const InputPhi_t& phi, real h, int samples)
	{
        //fast track
        int cubeindex = 0;
        if (phi.first.x <= 0) cubeindex |= 1;
        if (phi.first.y <= 0) cubeindex |= 2;
        if (phi.first.z <= 0) cubeindex |= 4;
        if (phi.first.w <= 0) cubeindex |= 8;
        if (phi.second.x <= 0) cubeindex |= 16;
        if (phi.second.y <= 0) cubeindex |= 32;
        if (phi.second.z <= 0) cubeindex |= 64;
        if (phi.second.w <= 0) cubeindex |= 128;
        if (cubeindex == 0b00000000)
            return { make_real4(0,0,0,0), make_real4(0,0,0,0) };
        if (cubeindex == 0b11111111)
            return { make_real4(0.125,0.125,0.125,0.125) * (h * h * h), make_real4(0.125,0.125,0.125,0.125) * (h * h * h) };

        //bounding box for better sampling
        std::vector<Eigen::Vector3d> points;
        if (utils::inside(phi.first.x )) points.push_back(Eigen::Vector3d(0, 0, 0));
        if (utils::inside(phi.first.y )) points.push_back(Eigen::Vector3d(1, 0, 0));
        if (utils::inside(phi.first.z )) points.push_back(Eigen::Vector3d(0, 1, 0));
        if (utils::inside(phi.first.w )) points.push_back(Eigen::Vector3d(1, 1, 0));
        if (utils::inside(phi.second.x)) points.push_back(Eigen::Vector3d(0, 0, 1));
        if (utils::inside(phi.second.y)) points.push_back(Eigen::Vector3d(1, 0, 1));
        if (utils::inside(phi.second.z)) points.push_back(Eigen::Vector3d(0, 1, 1));
        if (utils::inside(phi.second.w)) points.push_back(Eigen::Vector3d(1, 1, 1));
        if (utils::inside(phi.first.x) != utils::inside(phi.first.y)) points.push_back(Eigen::Vector3d(phi.first.x / (phi.first.x - phi.first.y), 0, 0));
        if (utils::inside(phi.first.x) != utils::inside(phi.first.z)) points.push_back(Eigen::Vector3d(0, phi.first.x / (phi.first.x - phi.first.z), 0));
        if (utils::inside(phi.first.z) != utils::inside(phi.first.w)) points.push_back(Eigen::Vector3d(phi.first.z / (phi.first.z - phi.first.w), 1, 0));
        if (utils::inside(phi.first.y) != utils::inside(phi.first.w)) points.push_back(Eigen::Vector3d(1, phi.first.y / (phi.first.y - phi.first.w), 0));
        if (utils::inside(phi.second.x) != utils::inside(phi.second.y)) points.push_back(Eigen::Vector3d(phi.second.x / (phi.second.x - phi.second.y), 0, 1));
        if (utils::inside(phi.second.x) != utils::inside(phi.second.z)) points.push_back(Eigen::Vector3d(0, phi.second.x / (phi.second.x - phi.second.z), 1));
        if (utils::inside(phi.second.z) != utils::inside(phi.second.w)) points.push_back(Eigen::Vector3d(phi.second.z / (phi.second.z - phi.second.w), 1, 1));
        if (utils::inside(phi.second.y) != utils::inside(phi.second.w)) points.push_back(Eigen::Vector3d(1, phi.second.y / (phi.second.y - phi.second.w), 1));
        if (utils::inside(phi.first.x) != utils::inside(phi.second.x)) points.push_back(Eigen::Vector3d(0, 0, phi.first.x / (phi.first.x - phi.second.x)));
        if (utils::inside(phi.first.y) != utils::inside(phi.second.y)) points.push_back(Eigen::Vector3d(1, 0, phi.first.y / (phi.first.y - phi.second.y)));
        if (utils::inside(phi.first.z) != utils::inside(phi.second.z)) points.push_back(Eigen::Vector3d(0, 1, phi.first.z / (phi.first.z - phi.second.z)));
        if (utils::inside(phi.first.w) != utils::inside(phi.second.w)) points.push_back(Eigen::Vector3d(1, 1, phi.first.w / (phi.first.w - phi.second.w)));

        ar::geom3d::AABBBox box = ar::geom3d::encloses(points);
        CI_LOG_V("AABB: min=" << box.min.transpose() << ", max=" << box.max.transpose());

        InterpolationWeight_t weights = { 0 };
        //now do the actual sampling
        for (int i=0; i<samples; ++i)
        {
            std::array<double, 3> xyz = utils::halton<3>(i);
            real x = xyz[0] * (box.max.x() - box.min.x()) + box.min.x();
            real y = xyz[1] * (box.max.y() - box.min.y()) + box.min.y();
            real z = xyz[2] * (box.max.z() - box.min.z()) + box.min.z();
            if (interpolate(phi, x, y, z) > 0) continue;
            weights.first.x += (1 - x)*(1 - y)*(1 - z);
            weights.first.y += x * (1 - y)*(1 - z);
            weights.first.z += (1 - x)*y*(1 - z);
            weights.first.w += x * y*(1 - z);
            weights.second.x += (1 - x)*(1 - y)*z;
            weights.second.y += x * (1 - y) * z;
            weights.second.z += (1 - x) * y * z;
            weights.second.w += x * y * z;
        }
        //scale it
        real scale = h * h * h * (box.max - box.min).prod() / samples;
        weights.first *= scale;
        weights.second *= scale;
        return weights;

#if 0
		static thread_local std::default_random_engine rnd;
		static thread_local std::uniform_real_distribution<real> distr(0, 1);
		InterpolationWeight_t weights = { 0 };
		for (int i=0; i<samples; ++i)
		{
			real x = distr(rnd);
			real y = distr(rnd);
			real z = distr(rnd);
			if (interpolate(phi, x, y, z) > 0) continue;
			weights.first.x += (1 - x)*(1 - y)*(1 - z);
			weights.first.y += x*(1 - y)*(1 - z);
			weights.first.z += (1 - x)*y*(1 - z);
			weights.first.w += x*y*(1 - z);
			weights.second.x += (1 - x)*(1 - y)*z;
			weights.second.y += x * (1 - y) * z;
			weights.second.z += (1 - x) * y * z;
			weights.second.w += x * y * z;
		}
		weights.first *= h*h*h / samples;
		weights.second *= h*h*h / samples;
		return weights;
#endif
	}

    Integration3D::InterpolationWeight_t Integration3D::volumeIntegralLinear(const InputPhi_t& phi, real h)
    {
#define EDGE_INTERSECTIONS                                    \
        /* Input: phi0 - phi8 */                              \
        real x01 = phi0 / (phi0 - phi1); if(x01==0) x01=1e-5; \
        real x23 = phi2 / (phi2 - phi3); if(x23==0) x23=1e-5; \
        real x02 = phi0 / (phi0 - phi2); if(x02==0) x02=1e-5; \
        real x13 = phi1 / (phi1 - phi3); if(x13==0) x13=1e-5; \
        real x45 = phi4 / (phi4 - phi5); if(x45==0) x45=1e-5; \
        real x67 = phi6 / (phi6 - phi7); if(x67==0) x67=1e-5; \
        real x46 = phi4 / (phi4 - phi6); if(x46==0) x46=1e-5; \
        real x57 = phi5 / (phi5 - phi7); if(x57==0) x57=1e-5; \
        real x04 = phi0 / (phi0 - phi4); if(x04==0) x04=1e-5; \
        real x15 = phi1 / (phi1 - phi5); if(x15==0) x15=1e-5; \
        real x26 = phi2 / (phi2 - phi6); if(x26==0) x26=1e-5; \
        real x37 = phi3 / (phi3 - phi7); if(x37==0) x37=1e-5

        const real phiArray[8] = {
            phi.first.x, phi.first.y, phi.first.z, phi.first.w,
            phi.second.x, phi.second.y, phi.second.z, phi.second.w
        };
        real weights[8] = {0,0,0,0,0,0,0,0};

#define CALL(fun, i0, i1, i2, i3, i4, i5, i6, i7)                                                                       \
        fun(                                                                                                            \
        phiArray[i0], phiArray[i1], phiArray[i2], phiArray[i3], phiArray[i4], phiArray[i5], phiArray[i6], phiArray[i7], \
        weights[i0], weights[i1], weights[i2], weights[i3], weights[i4], weights[i5], weights[i6], weights[i7]          \
        );
#define INV for (int i=0; i<8; ++i) weights[i] = 0.125 - weights[i];
#define CALL_INV(fun, i0, i1, i2, i3, i4, i5, i6, i7)                                                                   \
        fun(                                                                                                            \
        phiArray[i0], phiArray[i1], phiArray[i2], phiArray[i3], phiArray[i4], phiArray[i5], phiArray[i6], phiArray[i7], \
        weights[i0], weights[i1], weights[i2], weights[i3], weights[i4], weights[i5], weights[i6], weights[i7]          \
        );                                                                                                              \
        for (int i=0; i<8; ++i) weights[i] = 0.125 - weights[i];
#define SQ(v) ((v)*(v))
#define CUBE(v) ((v)*(v)*(v))

        const auto case0 = [](real phi0, real phi1, real phi2, real phi3, real phi4, real phi5, real phi6, real phi7,
                              real& w0,  real &w1,  real& w2,  real& w3,  real& w4,  real& w5,  real& w6,  real& w7)
        {
            //nothing filled, completely outside
            w0 = w1 = w2 = w3 = w4 = w5 = w6 = w7 = 0;
        };
        const auto case1 = [](real phi0, real phi1, real phi2, real phi3, real phi4, real phi5, real phi6, real phi7,
                              real& w0,  real &w1,  real& w2,  real& w3,  real& w4,  real& w5,  real& w6,  real& w7)
        {
            //everything filled, completely inside
            w0 = w1 = w2 = w3 = w4 = w5 = w6 = w7 = 1.0 / 8.0;
        };
        const auto case2 = [](real phi0, real phi1, real phi2, real phi3, real phi4, real phi5, real phi6, real phi7,
                              real& w0,  real &w1,  real& w2,  real& w3,  real& w4,  real& w5,  real& w6,  real& w7)
        {
            //lower corner is inside
            EDGE_INTERSECTIONS;
            w0 += (x01*x02*x04*(-8*(525 - 133*x04 + x02*(-147 + 31*x04)) + x01*(1176 - 248*x04 + x02*(-280 + 53*x04)))) / -20160;
            w1 += (SQ(x01)*x02*x04*(1176 - 248*x04 + x02*(-280 + 53*x04)))/20160;
            w2 += (x01*SQ(x02)*x04*(1176 - 248*x04 + x01*(-280 + 53*x04)))/20160;
            w3 += SQ(x01)*SQ(x02)*x04/72 - (53*SQ(x01)*SQ(x02)*SQ(x04))/20160;
            w4 += (x01*x02*(1064 - 248*x02 + x01*(-248 + 53*x02))*SQ(x04))/20160;
            w5 += (31*SQ(x01)*x02*SQ(x04))/2520 - (53*SQ(x01)*SQ(x02)*SQ(x04))/20160;
            w6 += (31*x01*SQ(x02)*SQ(x04))/2520 - (53*SQ(x01)*SQ(x02)*SQ(x04))/20160;
            w7 += (53*SQ(x01)*SQ(x02)*SQ(x04))/20160;
            assert(w0 >= 0 && w0 <= 0.125); assert(w1 >= 0 && w1 <= 0.125); assert(w2 >= 0 && w2 <= 0.125); assert(w3 >= 0 && w3 <= 0.125);
            assert(w4 >= 0 && w4 <= 0.125); assert(w5 >= 0 && w5 <= 0.125); assert(w6 >= 0 && w6 <= 0.125); assert(w7 >= 0 && w7 <= 0.125);
        };
        const auto case3 = [](real phi0, real phi1, real phi2, real phi3, real phi4, real phi5, real phi6, real phi7,
                              real& w0,  real &w1,  real& w2,  real& w3,  real& w4,  real& w5,  real& w6,  real& w7)
        {
            //lower edge is inside
            EDGE_INTERSECTIONS;
            w0 = (x13 * (SQ(x04) * (x13 - 12) + 2 * x04 * (30 + x13 * (x15 - 4) - 8 * x15) + 2 * (x13 * (x15 - 6) - 6
                * (x15 - 5)) * x15) + 2 * x02 * (2 * SQ(x04) * (x13 - 12) + 2 * x04 * (45 + x13 * (x15 - 6) - 6 * x15) +
                (30 + x13 * (x15 - 8) - 4 * x15) * x15) + SQ(x02) * (10 * SQ(x04) + 4 * x04 * (x15 - 12) + (x15 - 12) *
                x15)) / 1440;
            w1 += (2 * x02 * (SQ(x04) * (-6 + x13) + 2 * x04 * (15 + x13 * (-4 + x15) - 4 * x15) + 2 * (x13 * (-6 + x15) -
                    3 * (-5 + x15)) * x15) + SQ(x02) * (2 * SQ(x04) + 2 * x04 * (-6 + x15) + (-8 + x15) * x15) + x13 * (
                    SQ(x04) * (-8 + x13) + 4 * x04 * (15 + x13 * (-3 + x15) - 6 * x15) + 2 * x15 * (90 - 24 * x15 + x13
                        * (-24 + 5 * x15)))) / 1440.0;
            w2 += (-(SQ(x02) * (10 * SQ(x04) + 4 * x04 * (-12 + x15) + (-12 + x15) * x15)) - 2 * x02 * x13 * (2 * SQ(x04)
                    + 2 * x04 * (-6 + x15) + (-8 + x15) * x15) -
                SQ(x13) * (SQ(x04) + 2 * x04 * (-4 + x15) + 2 * (-6 + x15) * x15)) / 1440.;
            w3 += (-(SQ(x02) * (2 * SQ(x04) + 2 * x04 * (-6 + x15) + (-8 + x15) * x15)) - 2 * x02 * x13 * (SQ(x04) + 2 *
                    x04 * (-4 + x15) + 2 * (-6 + x15) * x15) -
                SQ(x13) * (SQ(x04) + 4 * x04 * (-3 + x15) + 2 * x15 * (-24 + 5 * x15))) / 1440.;
            w4 += (-(SQ(x04) * (2 * x02 * (-24 + 5 * x02) + 4 * (-3 + x02) * x13 + SQ(x13))) - 2 * x04 * (2 * (-6 + x02)
                    * x02 + 2 * (-4 + x02) * x13 + SQ(x13)) * x15 -
                (SQ(x02) + 2 * x02 * (-4 + x13) + 2 * (-6 + x13) * x13) * SQ(x15)) / 1440.;
            w5 += (-(SQ(x04) * (2 * (-6 + x02) * x02 + 2 * (-4 + x02) * x13 + SQ(x13))) - 2 * x04 * (SQ(x02) + 2 * x02 *
                    (-4 + x13) + 2 * (-6 + x13) * x13) * x15 -
                (SQ(x02) + 4 * x02 * (-3 + x13) + 2 * x13 * (-24 + 5 * x13)) * SQ(x15)) / 1440.;
            w6 += (SQ(x02) * SQ(x04)) / 144. + (x02 * SQ(x04) * x13) / 360. + (SQ(x04) * SQ(x13)) / 1440. + (SQ(x02) *
                    x04 * x15) / 360. + (x02 * x04 * x13 * x15) / 360. +
                (x04 * SQ(x13) * x15) / 720. + (SQ(x02) * SQ(x15)) / 1440. + (x02 * x13 * SQ(x15)) / 720. + (SQ(x13) *
                    SQ(x15)) / 720.;
            w7 += (SQ(x02) * SQ(x04)) / 720. + (x02 * SQ(x04) * x13) / 720. + (SQ(x04) * SQ(x13)) / 1440. + (SQ(x02) *
                    x04 * x15) / 720. + (x02 * x04 * x13 * x15) / 360. +
                (x04 * SQ(x13) * x15) / 360. + (SQ(x02) * SQ(x15)) / 1440. + (x02 * x13 * SQ(x15)) / 360. + (SQ(x13) *
                    SQ(x15)) / 144.;
            assert(w0 >= 0 && w0 <= 0.125); assert(w1 >= 0 && w1 <= 0.125); assert(w2 >= 0 && w2 <= 0.125); assert(w3 >= 0 && w3 <= 0.125);
            assert(w4 >= 0 && w4 <= 0.125); assert(w5 >= 0 && w5 <= 0.125); assert(w6 >= 0 && w6 <= 0.125); assert(w7 >= 0 && w7 <= 0.125);
        };
        const auto case4 = [](real phi0, real phi1, real phi2, real phi3, real phi4, real phi5, real phi6, real phi7,
                              real& w0,  real &w1,  real& w2,  real& w3,  real& w4,  real& w5,  real& w6,  real& w7)
        {
            //lower plane is inside
            EDGE_INTERSECTIONS;
            w0 += (-9 * SQ(x04) - 3 * SQ(x15) + 16 * x26 - 3 * SQ(x26) + 8 * x37 - 2 * x26 * x37 - SQ(x37) -
                2 * x15 * (-8 + x26 + x37) - 2 * x04 * (-16 + 3 * x15 + 3 * x26 + x37)) / 288;
            w1 += (-3 * SQ(x04) - 9 * SQ(x15) - (-8 + x26) * x26 - 2 * (-8 + x26) * x37 - 3 * SQ(x37) - 2 * x04 * (-8 + 3
                * x15 + x26 + x37) - 2 * x15 * (-16 + x26 + 3 * x37)) / 288.;
            w2 += (-3 * SQ(x04) - SQ(x15) + 32 * x26 - 9 * SQ(x26) + 16 * x37 - 6 * x26 * x37 - 3 * SQ(x37) - 2 * x15 * (
                -4 + x26 + x37) - 2 * x04 * (-8 + x15 + 3 * x26 + x37)) / 288.;
            w3 += (-SQ(x04) - 3 * SQ(x15) + 16 * x26 - 3 * SQ(x26) + 32 * x37 - 6 * x26 * x37 - 9 * SQ(x37) - 2 * x04 * (
                -4 + x15 + x26 + x37) - 2 * x15 * (-8 + x26 + 3 * x37)) / 288.;
            w4 += SQ(x04) / 32. + (x04 * x15) / 48. + SQ(x15) / 96. + (x04 * x26) / 48. + (x15 * x26) / 144. + SQ(x26) /
                96. + (x04 * x37) / 144. + (x15 * x37) / 144. + (x26 * x37) / 144. + SQ(x37) / 288.;
            w5 += SQ(x04) / 96. + (x04 * x15) / 48. + SQ(x15) / 32. + (x04 * x26) / 144. + (x15 * x26) / 144. + SQ(x26) /
                288. + (x04 * x37) / 144. + (x15 * x37) / 48. + (x26 * x37) / 144. + SQ(x37) / 96.;
            w6 += SQ(x04) / 96. + (x04 * x15) / 144. + SQ(x15) / 288. + (x04 * x26) / 48. + (x15 * x26) / 144. + SQ(x26)
                / 32. + (x04 * x37) / 144. + (x15 * x37) / 144. + (x26 * x37) / 48. + SQ(x37) / 96.;
            w7 += (SQ(x04) + 3 * SQ(x15) + 2 * x15 * x26 + 3 * SQ(x26) + 6 * x15 * x37 + 6 * x26 * x37 + 9 * SQ(x37) + 2
                * x04 * (x15 + x26 + x37)) / 288.;
            assert(w0 >= 0 && w0 <= 0.125); assert(w1 >= 0 && w1 <= 0.125); assert(w2 >= 0 && w2 <= 0.125); assert(w3 >= 0 && w3 <= 0.125);
            assert(w4 >= 0 && w4 <= 0.125); assert(w5 >= 0 && w5 <= 0.125); assert(w6 >= 0 && w6 <= 0.125); assert(w7 >= 0 && w7 <= 0.125);
        };
        const auto case5 = [](real phi0, real phi1, real phi2, real phi3, real phi4, real phi5, real phi6, real phi7,
                              real& w0,  real &w1,  real& w2,  real& w3,  real& w4,  real& w5,  real& w6,  real& w7)
        {
            //vertex 0,1,2 inside
            EDGE_INTERSECTIONS;
            w0 += (SQ(x23-1) * (SQ(x04) * (-38 - 8 * x13 + SQ(x13)) * SQ(x23-1) -
                    2 * x04 * (-1 + x23) * (66 - 2 * x15 * (5 + 19 * x23) + SQ(x13) * (-4 + x15 + x15 * x23) - 2 * x13 *
                        (-9 + x15 * (3 + 4 * x23))) +
                    x15 * (48 + 132 * x23 - x15 * (7 + 20 * x23 + 38 * SQ(x23)) + SQ(x13) * (-4 * (3 + 2 * x23) + x15 *
                            (2 + 2 * x23 + SQ(x23))) -
                        2 * x13 * (-2 * (11 + 9 * x23) + x15 * (5 + 6 * x23 + 4 * SQ(x23))))) +
                5 * x23 * (9 * SQ(x04) * (-2 + x23) * (2 + (-2 + x23) * x23) + x15 * x23 * (48 - 4 * (8 + 3 * x15) * x23
                        + 9 * x15 * SQ(x23)) -
                    2 * x04 * (x23 * (48 + 3 * x15 * (6 + x23 * (-8 + 3 * x23)) + x23 * (-16 + x26) - 4 * x26) + 6 * (-8
                        + x26)) + 2 * (12 + (-4 + x15 * (-2 + x23)) * x23) * x26 + (-4 + x23) * SQ(x26))) / 1440.;
            w1 += (-((-1 + x23) * (SQ(x04) * SQ(x23-1) * (SQ(x13) * (1 + x23) - 2 * x13 * (3 + 4 * x23) - 2 * (5 + 19 *
                        x23)) -
                    2 * x04 * (-1 + x23) * (24 + 66 * x23 - x15 * (7 + 20 * x23 + 38 * SQ(x23)) + SQ(x13) * (-6 - 4 *
                            x23 + x15 * (2 + 2 * x23 + SQ(x23))) -
                        2 * x13 * (-11 - 9 * x23 + x15 * (5 + 6 * x23 + 4 * SQ(x23)))) + x15 *
                    (52 + 96 * x23 + 132 * SQ(x23) - x15 * (11 + 21 * x23 + 30 * SQ(x23) + 38 * CUBE(x23)) +
                        SQ(x13) * (-8 * (6 + 3 * x23 + SQ(x23)) + x15 * (10 + 6 * x23 + 3 * SQ(x23) + CUBE(x23))) -
                        2 * x13 * (-2 * (39 + 22 * x23 + 9 * SQ(x23)) + x15 * (22 + 15 * x23 + 9 * SQ(x23) + 4 * CUBE(
                            x23)))))) -
                5 * SQ(x23) * (9 * SQ(x15) * SQ(x23) + 3 * SQ(x04) * (6 - 8 * x23 + 3 * SQ(x23)) + 2 * x15 * x23 * (-16
                        + x26) + (-8 + x26) * x26 -
                    2 * x04 * (24 + 9 * x15 * SQ(x23) - 2 * x26 + x23 * (-16 - 12 * x15 + x26)))) / 1440.;
            w2 += (-(SQ(x23-1) * (SQ(x04) * (10 + 4 * x13 + SQ(x13)) * SQ(x23-1) -
                    2 * x04 * (-1 + x23) * (SQ(x13) * (-4 + x15 + x15 * x23) + 2 * x13 * (-6 + x15 + 2 * x15 * x23) + 2
                        * (-12 + x15 + 5 * x15 * x23)) +
                    x15 * (-12 + x15 - 48 * x23 + 4 * x15 * x23 + 10 * x15 * SQ(x23) + 2 * x13 * (-8 + x15 - 12 * x23 +
                            2 * x15 * x23 + 2 * x15 * SQ(x23)) +
                        SQ(x13) * (-4 * (3 + 2 * x23) + x15 * (2 + 2 * x23 + SQ(x23)))))) +
                5 * x23 * (3 * SQ(x04) * (-2 + x23) * (2 + (-2 + x23) * x23) + x15 * x23 * (24 - 4 * (4 + x15) * x23 + 3
                        * x15 * SQ(x23)) + 2 * (24 + (-8 + x15 * (-2 + x23)) * x23) * x26 +
                    3 * (-4 + x23) * SQ(x26) - 2 * x04 * (6 * (-4 + x26) + x23 * (24 - 8 * x23 + x15 * (6 + x23 * (-8 +
                        3 * x23)) - 4 * x26 + x23 * x26)))) / 1440.;
            w3 += ((-1 + x23) * (SQ(x04) * SQ(x23-1) * (2 + 10 * x23 + SQ(x13) * (1 + x23) + x13 * (2 + 4 * x23)) -
                    2 * x04 * (-1 + x23) * (-6 + x15 - 24 * x23 + 4 * x15 * x23 + 10 * x15 * SQ(x23) + 2 * x13 * (-4 +
                            x15 - 6 * x23 + 2 * x15 * x23 + 2 * x15 * SQ(x23)) +
                        SQ(x13) * (-6 - 4 * x23 + x15 * (2 + 2 * x23 + SQ(x23)))) +
                    x15 * (-8 * (1 + 3 * x23 + 6 * SQ(x23)) + x15 * (1 + 3 * x23 + 6 * SQ(x23) + 10 * CUBE(x23)) +
                        SQ(x13) * (-8 * (6 + 3 * x23 + SQ(x23)) + x15 * (10 + 6 * x23 + 3 * SQ(x23) + CUBE(x23))) +
                        x13 * (-8 * (3 + 4 * x23 + 3 * SQ(x23)) + x15 * (4 + 6 * x23 + 6 * SQ(x23) + 4 * CUBE(x23))))) -
                5 * SQ(x23) * (3 * SQ(x15) * SQ(x23) + SQ(x04) * (6 + x23 * (-8 + 3 * x23)) + 2 * x15 * x23 * (-8 + x26)
                    + x26 * (-16 + 3 * x26) -
                    2 * x04 * (-2 * (-6 + x26) + x23 * (-8 + x15 * (-4 + 3 * x23) + x26)))) / 1440.;
            w4 += (-(SQ(x23-1) * (SQ(x04) * (-38 + (-8 + x13) * x13) * SQ(x23-1) +
                    SQ(x15) * (-7 + 2 * (-5 + x13) * x13 - 20 * x23 + 2 * (-6 + x13) * x13 * x23 + (-38 + (-8 + x13) *
                        x13) * SQ(x23)) -
                    2 * x04 * x15 * (-1 + x23) * (-2 * (5 + 19 * x23) + x13 * (-6 + x13 + (-8 + x13) * x23)))) +
                5 * x23 * (3 * SQ(x15) * (4 - 3 * x23) * SQ(x23) - 9 * SQ(x04) * (-2 + x23) * (2 + (-2 + x23) * x23) + 6
                    * x04 * x15 * x23 * (6 + x23 * (-8 + 3 * x23)) - 2 * x15 * (-2 + x23) * x23 * x26 +
                    2 * x04 * (6 + (-4 + x23) * x23) * x26 - (-4 + x23) * SQ(x26))) / 1440.;
            w5 += ((-1 + x23) * (SQ(x04) * SQ(x23-1) * (SQ(x13) * (1 + x23) - 2 * x13 * (3 + 4 * x23) - 2 * (5 + 19 * x23
                    )) -
                    2 * x04 * x15 * (-1 + x23) * (-7 - 20 * x23 - 38 * SQ(x23) + SQ(x13) * (2 + 2 * x23 + SQ(x23)) - 2 *
                        x13 * (5 + 6 * x23 + 4 * SQ(x23))) +
                    SQ(x15) * (-11 - 21 * x23 - 30 * SQ(x23) - 38 * CUBE(x23) + SQ(x13) * (10 + 6 * x23 + 3 * SQ(x23) +
                            CUBE(x23)) -
                        2 * x13 * (22 + 15 * x23 + 9 * SQ(x23) + 4 * CUBE(x23)))) + 5 * SQ(x23) *
                (9 * SQ(x15) * SQ(x23) + 3 * SQ(x04) * (6 - 8 * x23 + 3 * SQ(x23)) + 2 * x15 * x23 * x26 + SQ(x26) - 2 *
                    x04 * (3 * x15 * x23 * (-4 + 3 * x23) + (-2 + x23) * x26))) / 1440.;
            w6 += (SQ(x23-1) * (SQ(x04) * (10 + x13 * (4 + x13)) * SQ(x23-1) + SQ(x15) * (1 + 2 * x13 * (1 + x13) + 4 *
                        x23 + 2 * x13 * (2 + x13) * x23 + (10 + x13 * (4 + x13)) * SQ(x23)) -
                    2 * x04 * x15 * (-1 + x23) * (2 + 10 * x23 + x13 * (2 + x13 + (4 + x13) * x23))) +
                5 * x23 * (SQ(x15) * (4 - 3 * x23) * SQ(x23) - 3 * SQ(x04) * (-2 + x23) * (2 + (-2 + x23) * x23) - 2 *
                    x15 * (-2 + x23) * x23 * x26 - 3 * (-4 + x23) * SQ(x26) +
                    2 * x04 * (x15 * x23 * (6 + x23 * (-8 + 3 * x23)) + (6 + (-4 + x23) * x23) * x26))) / 1440.;
            w7 += (-((-1 + x23) * (SQ(x04) * SQ(x23-1) * (2 + 10 * x23 + SQ(x13) * (1 + x23) + x13 * (2 + 4 * x23)) -
                    2 * x04 * x15 * (-1 + x23) * (1 + 4 * x23 + 10 * SQ(x23) + SQ(x13) * (2 + 2 * x23 + SQ(x23)) + x13 *
                        (2 + 4 * x23 + 4 * SQ(x23))) +
                    SQ(x15) * (1 + 3 * x23 + 6 * SQ(x23) + 10 * CUBE(x23) + SQ(x13) * (10 + 6 * x23 + 3 * SQ(x23) + CUBE
                            (x23)) +
                        x13 * (4 + 6 * x23 + 6 * SQ(x23) + 4 * CUBE(x23))))) + 5 * SQ(x23) *
                (3 * SQ(x15) * SQ(x23) + SQ(x04) * (6 - 8 * x23 + 3 * SQ(x23)) + 2 * x15 * x23 * x26 + 3 * SQ(x26) + x04
                    * (2 * x15 * (4 - 3 * x23) * x23 - 2 * (-2 + x23) * x26))) / 1440.;
            assert(w0 >= 0 && w0 <= 0.125); assert(w1 >= 0 && w1 <= 0.125); assert(w2 >= 0 && w2 <= 0.125); assert(w3 >= 0 && w3 <= 0.125);
            assert(w4 >= 0 && w4 <= 0.125); assert(w5 >= 0 && w5 <= 0.125); assert(w6 >= 0 && w6 <= 0.125); assert(w7 >= 0 && w7 <= 0.125);
        };
        const auto case6 = [](real phi0, real phi1, real phi2, real phi3, real phi4, real phi5, real phi6, real phi7,
                              real& w0,  real &w1,  real& w2,  real& w3,  real& w4,  real& w5,  real& w6,  real& w7)
        {
            //vertex 0,1,2,4 inside
            EDGE_INTERSECTIONS;
            w0 += ((x15 * (-56 * x13 * (6 + 5 * (-3 + x13) * x13) + (-36 + x13 * (150 + 53 * (-4 + x13) * x13)) * x15) *
                    SQ(x23-1)) / x13 +
                (CUBE(x23) * (-252 - 176 * x26 + 53 * SQ(x26)) + 36 * SQ(x26) * (-2 + x45) * SQ(x45) - 6 * x23 * x26 *
                    x45 * (56 - 24 * x45 + 5 * x26 * (-8 + 3 * x45)) +
                    4 * SQ(x23) * (x26 * (168 - 36 * x45) + 42 * (7 - 4 * x45) + SQ(x26) * (-62 + 9 * x45))) / x23 +
                (2 * (-1567 - 36 * CUBE(x23) + 6 * SQ(x23) * (-17 + 35 * x45) + 12 * x23 * (75 + 7 * x45 * (-19 + 7 *
                        x45)) + 7 * x45 * (543 + x45 * (-429 + 115 * x45)) -
                    3 * SQ(x15) * (-33 + 12 * CUBE(x23) + 4 * x23 * (2 + 7 * x45) - 2 * SQ(x23) * (11 + 7 * x45) + 7 *
                        x45 * (13 + 5 * (-3 + x45) * x45)) +
                    2 * x15 * (-85 + 36 * CUBE(x23) + 18 * SQ(x23) * (1 - 7 * x45) + 36 * x23 * (-4 + 7 * x45) + 7 * x45
                        * (57 + 25 * (-3 + x45) * x45)))) / (-1 + x45) + 5040 * x45 * x46 -
                1680 * SQ(x45) * x46 - 1680 * x45 * SQ(x46) + 420 * SQ(x45) * SQ(x46) +
                (x45 * (x45 * (701 - 1371 * x46 + 813 * SQ(x46) - 179 * CUBE(x46)) + 24 * (-92 + 209 * x46 - 155 * SQ(
                        x46) + 41 * CUBE(x46)) +
                    SQ(x26) * (8 * (25 - 87 * x46 + 102 * SQ(x46) - 31 * CUBE(x46)) + x45 * (-35 + 141 * x46 - 195 * SQ(
                        x46) + 53 * CUBE(x46))) +
                    x26 * (-6 * x45 * (-29 + 131 * x46 - 149 * SQ(x46) + 35 * CUBE(x46)) + 8 * (-85 + 321 * x46 - 351 *
                        SQ(x46) + 97 * CUBE(x46))))) / (-1 + x46)) / 20160.;
            w1 += (252 * SQ(x23) - (x15 * (-1 + x23) * (-224 * x13 * (3 + 4 * (-3 + x13) * x13) - 60 * x15 + 15 * x13 * (
                        22 + 13 * (-4 + x13) * x13) * x15 +
                    (-56 * x13 * (6 + 5 * (-3 + x13) * x13) + (-36 + x13 * (150 + 53 * (-4 + x13) * x13)) * x15) * x23))
                / x13 + 176 * SQ(x23) * x26 - 53 * SQ(x23) * SQ(x26) + 672 * x23 * x45 +
                144 * x23 * x26 * x45 - 36 * x23 * SQ(x26) * x45 - 144 * x26 * SQ(x45) + 90 * SQ(x26) * SQ(x45) - (36 *
                    SQ(x26) * CUBE(x45)) / x23 -
                (2 * (197 - 36 * CUBE(x23) + 693 * x45 - 1743 * SQ(x45) + 805 * CUBE(x45) + 6 * SQ(x23) * (-31 + 35 *
                        x45) + 12 * x23 * (40 - 84 * x45 + 49 * SQ(x45)) -
                    3 * SQ(x15) * (51 + 12 * CUBE(x23) + x23 * (36 - 56 * x45) + SQ(x23) * (6 - 14 * x45) - 105 * x45 +
                        35 * SQ(x45) + 35 * CUBE(x45)) +
                    x15 * (418 + 72 * CUBE(x23) + SQ(x23) * (204 - 252 * x45) - 882 * x45 + 210 * SQ(x45) + 350 * CUBE(
                        x45) - 24 * x23 * (-9 + 14 * x45)))) / (-1 + x45) +
                1680 * SQ(x45) * x46 - 420 * SQ(x45) * SQ(x46) + (SQ(x45) *
                    (-701 + x46 * (1371 + x46 * (-813 + 179 * x46)) + SQ(x26) * (35 + x46 * (-141 + (195 - 53 * x46) *
                        x46)) + 6 * x26 * (-29 + x46 * (131 + x46 * (-149 + 35 * x46))))) / (-1 + x46)) / 20160.;
            w2 += (-((x15 * (x13 * (336 - 90 * x15) + 36 * x15 + 12 * SQ(x13) * (-28 + 3 * x15) + CUBE(x13) * (-280 + 53
                    * x15)) * SQ(x23-1)) / x13) +
                (2 * (-1457 - 48 * CUBE(x23) + 3087 * x45 - 2037 * SQ(x45) + 455 * CUBE(x45) + 18 * SQ(x23) * (-13 + 21
                        * x45) + 12 * x23 * (107 - 175 * x45 + 56 * SQ(x45)) +
                    SQ(x15) * (41 - 48 * CUBE(x23) - 63 * x45 + 105 * SQ(x45) - 35 * CUBE(x45) - 12 * x23 * (5 + 7 * x45
                    ) + 6 * SQ(x23) * (17 + 7 * x45)) +
                    6 * x15 * (19 + 16 * CUBE(x23) + SQ(x23) * (22 - 70 * x45) + 35 * x45 - 105 * SQ(x45) + 35 * CUBE(
                        x45) + 4 * x23 * (-23 + 35 * x45)))) / (-1 + x45) +
                (3 * (5 * CUBE(x23) * (-28 + x26 * (-48 + 13 * x26)) + 4 * SQ(x26) * SQ(x45) * (-8 + 3 * x45) + 2 * x23
                    * x26 * x45 * (-112 + 72 * x26 + 32 * x45 - 17 * x26 * x45) +
                    4 * SQ(x23) * (112 - 42 * x45 + x26 * (196 - 16 * x45 + x26 * (-68 + 3 * x45))))) / x23 - 420 * (-4
                    + x45) * x45 * SQ(x46) -
                (x45 * (x45 * (-227 + 53 * x46 + 389 * SQ(x46) - 179 * CUBE(x46)) + 8 * (116 - 31 * x46 - 220 * SQ(x46)
                        + 123 * CUBE(x46)) +
                    SQ(x26) * (-8 * (45 - 95 * x46 + 31 * SQ(x46) + 31 * CUBE(x46)) + x45 * (45 - 115 * x46 + 53 * SQ(
                        x46) + 53 * CUBE(x46))) -
                    2 * x26 * (-4 * (118 - 253 * x46 + 62 * SQ(x46) + 97 * CUBE(x46)) + x45 * (77 - 199 * x46 + 53 * SQ(
                        x46) + 105 * CUBE(x46))))) / (-1 + x46)) / 20160.;
            w3 += (420 * SQ(x23) + (x15 * (-1 + x23) * (36 * x15 * (1 + x23) + 12 * SQ(x13) * (-28 + 3 * x15) * (1 + x23)
                    - 6 * x13 * (-56 * (1 + x23) + x15 * (17 + 15 * x23)) +
                    CUBE(x13) * (-56 * (16 + 5 * x23) + x15 * (195 + 53 * x23)))) / x13 + 720 * SQ(x23) * x26 - 195 * SQ
                (x23) * SQ(x26) + 504 * x23 * x45 + 192 * x23 * x26 * x45 -
                36 * x23 * SQ(x26) * x45 - 192 * x26 * SQ(x45) + 102 * SQ(x26) * SQ(x45) - (36 * SQ(x26) * CUBE(x45)) /
                x23 +
                (2 * (169 + 48 * CUBE(x23) - 1015 * x45 + 1337 * SQ(x45) - 455 * CUBE(x45) - 6 * SQ(x23) * (-53 + 63 *
                        x45) - 12 * x23 * (65 - 119 * x45 + 56 * SQ(x45)) -
                    2 * x15 * (71 - 24 * x23 + 48 * CUBE(x23) - 175 * x45 + 35 * SQ(x45) + 105 * CUBE(x45) - 30 * SQ(x23
                    ) * (-5 + 7 * x45)) +
                    SQ(x15) * (15 + 48 * CUBE(x23) + x23 * (60 - 84 * x45) - 49 * x45 + 35 * SQ(x45) + 35 * CUBE(x45) -
                        6 * SQ(x23) * (3 + 7 * x45)))) / (-1 + x45) +
                420 * SQ(x45) * SQ(x46) + (SQ(x45) * (-227 + x46 * (53 + (389 - 179 * x46) * x46) + SQ(x26) * (45 + x46
                        * (-115 + 53 * x46 * (1 + x46))) -
                    2 * x26 * (77 + x46 * (-199 + x46 * (53 + 105 * x46))))) / (-1 + x46)) / 20160.;
            w4 += (-(((-36 + x13 * (150 + 53 * (-4 + x13) * x13)) * SQ(x15) * SQ(x23-1)) / x13) + 504 * x23 - 84 * SQ(x23
                ) + 504 * x23 * x26 - 104 * SQ(x23) * x26 + 248 * x23 * SQ(x26) -
                53 * SQ(x23) * SQ(x26) - 336 * x23 * x45 - 336 * x26 * x45 - 192 * x23 * x26 * x45 - 240 * SQ(x26) * x45
                - 36 * x23 * SQ(x26) * x45 + 192 * x26 * SQ(x45) +
                90 * SQ(x26) * SQ(x45) + (72 * SQ(x26) * SQ(x45)) / x23 - (36 * SQ(x26) * CUBE(x45)) / x23 +
                (6 * (-2 * x15 * (9 + 12 * CUBE(x23) + x23 * (64 - 28 * x45) - 91 * x45 + 105 * SQ(x45) - 35 * CUBE(x45)
                        + 2 * SQ(x23) * (-25 + 7 * x45)) +
                    SQ(x15) * (-33 + 12 * CUBE(x23) + 91 * x45 - 105 * SQ(x45) + 35 * CUBE(x45) + 4 * x23 * (2 + 7 * x45
                    ) - 2 * SQ(x23) * (11 + 7 * x45)) +
                    3 * (-81 + 4 * CUBE(x23) + 175 * x45 - 133 * SQ(x45) + 35 * CUBE(x45) + 2 * SQ(x23) * (-13 + 7 * x45
                    ) + 4 * x23 * (17 - 21 * x45 + 7 * SQ(x45))))) / (-1 + x45) +
                5040 * x45 * x46 - 1680 * SQ(x45) * x46 - 1680 * x45 * SQ(x46) + 420 * SQ(x45) * SQ(x46) -
                (x45 * (3 * (x45 * (-65 + 103 * x46 - 65 * SQ(x46) + 15 * CUBE(x46)) - 8 * (-34 + 71 * x46 - 55 * SQ(x46
                    ) + 15 * CUBE(x46))) +
                    2 * x26 * (8 * (31 - 102 * x46 + 87 * SQ(x46) - 25 * CUBE(x46)) + x45 * (-53 + 195 * x46 - 141 * SQ(
                        x46) + 35 * CUBE(x46))) +
                    SQ(x26) * (8 * (25 - 87 * x46 + 102 * SQ(x46) - 31 * CUBE(x46)) + x45 * (-35 + 141 * x46 - 195 * SQ(
                        x46) + 53 * CUBE(x46))))) / (-1 + x46)) / 20160.;
            w5 += (84 * SQ(x23) + (SQ(x15) * (-1 + x23) * (-12 * (5 + 3 * x23) + x13 * (30 * (11 + 5 * x23) + (-4 + x13)
                    * x13 * (195 + 53 * x23)))) / x13 + 104 * SQ(x23) * x26 +
                53 * SQ(x23) * SQ(x26) + 336 * x23 * x45 + 192 * x23 * x26 * x45 + 36 * x23 * SQ(x26) * x45 - 192 * x26
                * SQ(x45) - 90 * SQ(x26) * SQ(x45) +
                (36 * SQ(x26) * CUBE(x45)) / x23 - (6 * (9 + 12 * CUBE(x23) + 161 * x45 - 259 * SQ(x45) + 105 * CUBE(x45
                    ) + SQ(x23) * (-50 + 42 * x45) +
                    4 * x23 * (16 - 42 * x45 + 21 * SQ(x45)) + SQ(x15) * (51 + 12 * CUBE(x23) + x23 * (36 - 56 * x45) +
                        SQ(x23) * (6 - 14 * x45) - 105 * x45 + 35 * SQ(x45) +
                        35 * CUBE(x45)) - 2 * x15 * (-33 + 12 * CUBE(x23) + x23 * (8 - 28 * x45) + 49 * x45 + 35 * SQ(
                        x45) - 35 * CUBE(x45) + 2 * SQ(x23) * (-11 + 7 * x45)))) / (-1 + x45) +
                1680 * SQ(x45) * x46 - 420 * SQ(x45) * SQ(x46) + (SQ(x45) *
                    (-195 - x26 * (106 + 35 * x26) + 309 * x46 + 3 * x26 * (130 + 47 * x26) * x46 - 3 * (65 + x26 * (94
                        + 65 * x26)) * SQ(x46) + (45 + x26 * (70 + 53 * x26)) * CUBE(x46))) / (-1 + x46)) / 20160.;
            w6 += (((36 - 90 * x13 + 36 * SQ(x13) + 53 * CUBE(x13)) * SQ(x15) * SQ(x23-1)) / x13 -
                (CUBE(x23) * (84 + 176 * x26 + 195 * SQ(x26)) + 12 * SQ(x26) * SQ(x45) * (-8 + 3 * x45) - 6 * x23 * x26
                    * x45 * (-56 - 72 * x26 + 24 * x45 + 17 * x26 * x45) +
                    12 * SQ(x23) * (14 * (-2 + x45) + SQ(x26) * (-68 + 3 * x45) + 4 * x26 * (-14 + 3 * x45))) / x23 +
                (2 * (-2 * x15 * (-55 + 48 * CUBE(x23) + x23 * (228 - 84 * x45) - 63 * x45 + 105 * SQ(x45) - 35 * CUBE(
                        x45) + 6 * SQ(x23) * (-31 + 7 * x45)) +
                    SQ(x15) * (-41 + 48 * CUBE(x23) + 63 * x45 - 105 * SQ(x45) + 35 * CUBE(x45) + 12 * x23 * (5 + 7 *
                        x45) - 6 * SQ(x23) * (17 + 7 * x45)) +
                    3 * (-149 + 16 * CUBE(x23) + 259 * x45 - 161 * SQ(x45) + 35 * CUBE(x45) + 6 * SQ(x23) * (-15 + 7 *
                        x45) + 4 * x23 * (47 - 49 * x45 + 14 * SQ(x45))))) / (-1 + x45) -
                420 * (-4 + x45) * x45 * SQ(x46) + (x45 * (x45 * (53 + 53 * x46 - 115 * SQ(x46) + 45 * CUBE(x46)) - 8 *
                    (31 + 31 * x46 - 95 * SQ(x46) + 45 * CUBE(x46)) +
                    2 * x26 * (1 + x46) * (-8 * (25 - 56 * x46 + 25 * SQ(x46)) + x45 * (35 - 88 * x46 + 35 * SQ(x46))) +
                    SQ(x26) * (-8 * (45 - 95 * x46 + 31 * SQ(x46) + 31 * CUBE(x46)) + x45 * (45 - 115 * x46 + 53 * SQ(
                        x46) + 53 * CUBE(x46))))) / (-1 + x46)) / 20160.;
            w7 += (84 * SQ(x23) - (SQ(x15) * (-1 + x23) * (36 * (1 + x23) + 36 * SQ(x13) * (1 + x23) - 6 * x13 * (17 + 15
                    * x23) + CUBE(x13) * (195 + 53 * x23))) / x13 + 176 * SQ(x23) * x26 +
                195 * SQ(x23) * SQ(x26) + 168 * x23 * x45 + 144 * x23 * x26 * x45 + 36 * x23 * SQ(x26) * x45 - 144 * x26
                * SQ(x45) - 102 * SQ(x26) * SQ(x45) +
                (36 * SQ(x26) * CUBE(x45)) / x23 - (2 * (-55 + 48 * CUBE(x23) + 329 * x45 - 343 * SQ(x45) + 105 * CUBE(
                        x45) + 6 * SQ(x23) * (-31 + 21 * x45) +
                    12 * x23 * (19 - 35 * x45 + 14 * SQ(x45)) - 2 * x15 * (-41 + 48 * CUBE(x23) + x23 * (60 - 84 * x45)
                        + 77 * x45 + 35 * SQ(x45) - 35 * CUBE(x45) +
                        6 * SQ(x23) * (-17 + 7 * x45)) + SQ(x15) * (15 + 48 * CUBE(x23) + x23 * (60 - 84 * x45) - 49 *
                        x45 + 35 * SQ(x45) + 35 * CUBE(x45) - 6 * SQ(x23) * (3 + 7 * x45)))) /
                (-1 + x45) + 420 * SQ(x45) * SQ(x46) - (SQ(x45) * (53 + x46 * (53 + 5 * x46 * (-23 + 9 * x46)) + 2 * x26
                    * (1 + x46) * (35 + x46 * (-88 + 35 * x46)) +
                    SQ(x26) * (45 + x46 * (-115 + 53 * x46 * (1 + x46))))) / (-1 + x46)) / 20160.;
            assert(w0 >= 0 && w0 <= 0.125); assert(w1 >= 0 && w1 <= 0.125); assert(w2 >= 0 && w2 <= 0.125); assert(w3 >= 0 && w3 <= 0.125);
            assert(w4 >= 0 && w4 <= 0.125); assert(w5 >= 0 && w5 <= 0.125); assert(w6 >= 0 && w6 <= 0.125); assert(w7 >= 0 && w7 <= 0.125);
        };

        int cubeindex = 0;
        if (phiArray[0] <= 0) cubeindex |= 1;
        if (phiArray[1] <= 0) cubeindex |= 2;
        if (phiArray[2] <= 0) cubeindex |= 4;
        if (phiArray[3] <= 0) cubeindex |= 8;
        if (phiArray[4] <= 0) cubeindex |= 16;
        if (phiArray[5] <= 0) cubeindex |= 32;
        if (phiArray[6] <= 0) cubeindex |= 64;
        if (phiArray[7] <= 0) cubeindex |= 128;

        switch (cubeindex)
        {
        case 0b00000000: CALL(case0, 0, 1, 2, 3, 4, 5, 6, 7); break;

        case 0b00000001: CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7); break;
        case 0b00000010: CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6); break;
        case 0b00000100: CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5); break;
        case 0b00001000: CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4); break;
        case 0b00010000: CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3); break;
        case 0b00100000: CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2); break;
        case 0b01000000: CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1); break;
        case 0b10000000: CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0); break;

        case 0b11111110: CALL_INV(case2, 0, 1, 2, 3, 4, 5, 6, 7); break;
        case 0b11111101: CALL_INV(case2, 1, 3, 0, 2, 5, 7, 4, 6); break;
        case 0b11111011: CALL_INV(case2, 2, 0, 3, 1, 6, 4, 7, 5); break;
        case 0b11110111: CALL_INV(case2, 3, 2, 1, 0, 7, 6, 5, 4); break;
        case 0b11101111: CALL_INV(case2, 4, 5, 0, 1, 6, 7, 2, 3); break;
        case 0b11011111: CALL_INV(case2, 5, 1, 4, 0, 7, 3, 6, 2); break;
        case 0b10111111: CALL_INV(case2, 6, 4, 2, 0, 7, 5, 3, 1); break;
        case 0b01111111: CALL_INV(case2, 7, 6, 3, 2, 5, 4, 1, 0); break;

        case 0b00000011: CALL(case3, 0, 1, 2, 3, 4, 5, 6, 7); break;
        case 0b00001010: CALL(case3, 1, 3, 0, 2, 5, 7, 4, 6); break;
        case 0b00001100: CALL(case3, 3, 2, 1, 0, 7, 6, 5, 4); break;
        case 0b00000101: CALL(case3, 2, 0, 3, 1, 6, 4, 7, 5); break;
        case 0b00110000: CALL(case3, 4, 5, 0, 1, 6, 7, 2, 3); break;
        case 0b10100000: CALL(case3, 5, 7, 1, 3, 4, 6, 0, 2); break;
        case 0b11000000: CALL(case3, 7, 6, 3, 2, 5, 4, 1, 0); break;
        case 0b01010000: CALL(case3, 6, 4, 2, 0, 7, 5, 3, 1); break;
        case 0b00010001: CALL(case3, 4, 0, 6, 2, 5, 1, 7, 4); break;
        case 0b00100010: CALL(case3, 5, 1, 4, 0, 7, 3, 6, 2); break;
        case 0b01000100: CALL(case3, 6, 2, 7, 3, 4, 0, 5, 1); break;
        case 0b10001000: CALL(case3, 7, 3, 5, 1, 6, 2, 4, 0); break;

        case 0b11111100: CALL_INV(case3, 0, 1, 2, 3, 4, 5, 6, 7); break;
        case 0b11110101: CALL_INV(case3, 1, 3, 0, 2, 5, 7, 4, 6); break;
        case 0b11110011: CALL_INV(case3, 3, 2, 1, 0, 7, 6, 5, 4); break;
        case 0b11111010: CALL_INV(case3, 2, 0, 3, 1, 6, 4, 7, 5); break;
        case 0b11001111: CALL_INV(case3, 4, 5, 0, 1, 6, 7, 2, 3); break;
        case 0b01011111: CALL_INV(case3, 5, 7, 1, 3, 4, 6, 0, 2); break;
        case 0b00111111: CALL_INV(case3, 7, 6, 3, 2, 5, 4, 1, 0); break;
        case 0b10101111: CALL_INV(case3, 6, 4, 2, 0, 7, 5, 3, 1); break;
        case 0b11101110: CALL_INV(case3, 4, 0, 6, 2, 5, 1, 7, 4); break;
        case 0b11011101: CALL_INV(case3, 5, 1, 4, 0, 7, 3, 6, 2); break;
        case 0b10111011: CALL_INV(case3, 6, 2, 7, 3, 4, 0, 5, 1); break;
        case 0b01110111: CALL_INV(case3, 7, 3, 5, 1, 6, 2, 4, 0); break;

        case 0b00001111: CALL(case4, 0, 1, 2, 3, 4, 5, 6, 7); break;
        case 0b11110000: CALL(case4, 5, 4, 7, 6, 1, 0, 3, 2); break;
        case 0b00110011: CALL(case4, 4, 5, 0, 1, 6, 7, 2, 3); break;
        case 0b10101010: CALL(case4, 5, 7, 1, 3, 4, 6, 0, 2); break;
        case 0b11001100: CALL(case4, 7, 6, 3, 2, 5, 4, 1, 0); break;
        case 0b01010101: CALL(case4, 6, 4, 2, 0, 7, 5, 3, 1); break;

        case 0b00000111: CALL(case5, 0, 1, 2, 3, 4, 5, 6, 7); break;
        case 0b00010011: CALL(case5, 0, 4, 1, 5, 2, 6, 3, 7); break;
        case 0b00010101: CALL(case5, 0, 2, 4, 6, 1, 3, 5, 7); break;
        case 0b00001011: CALL(case5, 1, 3, 0, 2, 5, 7, 4, 6); break;
        case 0b00100011: CALL(case5, 1, 0, 5, 4, 3, 2, 7, 6); break;
        case 0b00101010: CALL(case5, 1, 5, 3, 7, 0, 4, 2, 6); break;
        case 0b00001101: CALL(case5, 2, 0, 3, 1, 6, 4, 7, 5); break;
        case 0b01000101: CALL(case5, 2, 6, 0, 4, 3, 7, 1, 5); break;
        case 0b01001100: CALL(case5, 2, 3, 6, 7, 0, 1, 4, 5); break;
        case 0b00001110: CALL(case5, 3, 2, 1, 0, 7, 6, 5, 4); break;
        case 0b10001010: CALL(case5, 3, 1, 7, 5, 2, 0, 6, 4); break;
        case 0b10001100: CALL(case5, 3, 7, 2, 6, 1, 5, 0, 4); break;
        case 0b01110000: CALL(case5, 4, 6, 5, 7, 0, 2, 1, 3); break;
        case 0b00110001: CALL(case5, 4, 5, 0, 1, 6, 7, 2, 3); break;
        case 0b01010001: CALL(case5, 4, 0, 6, 2, 5, 1, 7, 3); break;
        case 0b10110000: CALL(case5, 5, 4, 7, 6, 1, 0, 3, 2); break;
        case 0b00110010: CALL(case5, 5, 1, 4, 0, 7, 3, 6, 2); break;
        case 0b10100010: CALL(case5, 5, 7, 1, 3, 4, 6, 0, 2); break;
        case 0b11010000: CALL(case5, 6, 7, 4, 5, 2, 3, 0, 1); break;
        case 0b01010100: CALL(case5, 6, 4, 2, 0, 7, 5, 3, 1); break;
        case 0b11000100: CALL(case5, 6, 2, 7, 3, 4, 0, 5, 1); break;
        case 0b11100000: CALL(case5, 7, 5, 6, 4, 3, 1, 2, 0); break;
        case 0b10101000: CALL(case5, 7, 3, 5, 1, 6, 2, 4, 0); break;
        case 0b11001000: CALL(case5, 7, 6, 3, 2, 5, 4, 1, 0); break;

        case 0b11111000: CALL_INV(case5, 0, 1, 2, 3, 4, 5, 6, 7); break;
        case 0b11101100: CALL_INV(case5, 0, 4, 1, 5, 2, 6, 3, 7); break;
        case 0b11101010: CALL_INV(case5, 0, 2, 4, 6, 1, 3, 5, 7); break;
        case 0b11110100: CALL_INV(case5, 1, 3, 0, 2, 5, 7, 4, 6); break;
        case 0b11011100: CALL_INV(case5, 1, 0, 5, 4, 3, 2, 7, 6); break;
        case 0b11010101: CALL_INV(case5, 1, 5, 3, 7, 0, 4, 2, 6); break;
        case 0b11110010: CALL_INV(case5, 2, 0, 3, 1, 6, 4, 7, 5); break;
        case 0b10111010: CALL_INV(case5, 2, 6, 0, 4, 3, 7, 1, 5); break;
        case 0b10110011: CALL_INV(case5, 2, 3, 6, 7, 0, 1, 4, 5); break;
        case 0b11110001: CALL_INV(case5, 3, 2, 1, 0, 7, 6, 5, 4); break;
        case 0b01110101: CALL_INV(case5, 3, 1, 7, 5, 2, 0, 6, 4); break;
        case 0b01110011: CALL_INV(case5, 3, 7, 2, 6, 1, 5, 0, 4); break;
        case 0b10001111: CALL_INV(case5, 4, 6, 5, 7, 0, 2, 1, 3); break;
        case 0b11001110: CALL_INV(case5, 4, 5, 0, 1, 6, 7, 2, 3); break;
        case 0b10101110: CALL_INV(case5, 4, 0, 6, 2, 5, 1, 7, 3); break;
        case 0b01001111: CALL_INV(case5, 5, 4, 7, 6, 1, 0, 3, 2); break;
        case 0b11001101: CALL_INV(case5, 5, 1, 4, 0, 7, 3, 6, 2); break;
        case 0b01011101: CALL_INV(case5, 5, 7, 1, 3, 4, 6, 0, 2); break;
        case 0b00101111: CALL_INV(case5, 6, 7, 4, 5, 2, 3, 0, 1); break;
        case 0b10101011: CALL_INV(case5, 6, 4, 2, 0, 7, 5, 3, 1); break;
        case 0b00111011: CALL_INV(case5, 6, 2, 7, 3, 4, 0, 5, 1); break;
        case 0b00011111: CALL_INV(case5, 7, 5, 6, 4, 3, 1, 2, 0); break;
        case 0b01010111: CALL_INV(case5, 7, 3, 5, 1, 6, 2, 4, 0); break;
        case 0b00110111: CALL_INV(case5, 7, 6, 3, 2, 5, 4, 1, 0); break;
        
        //case 0b00010111: CALL(case6, 0, 1, 2, 3, 4, 5, 6, 7); break;
        //case 0b00101011: CALL(case6, 1, 3, 0, 2, 5, 7, 4, 6); break;
        //case 0b01001101: CALL(case6, 2, 0, 3, 1, 6, 4, 7, 5); break;
        //case 0b10001110: CALL(case6, 3, 2, 1, 0, 7, 6, 5, 4); break;
        //case 0b01110001: CALL(case6, 4, 5, 0, 1, 6, 7, 2, 3); break;
        //case 0b10110010: CALL(case6, 5, 7, 1, 3, 4, 6, 0, 2); break;
        //case 0b11010100: CALL(case6, 6, 7, 4, 5, 2, 3, 0, 1); break;
        //case 0b11101000: CALL(case6, 7, 6, 3, 2, 5, 4, 1, 0); break;

            //COMBINED

        case 0b00001001: 
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            break;
        case 0b00000110:
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            break;
        case 0b10010000:
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            break;
        case 0b01100000:
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b00100001:
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            break;
        case 0b00010010:
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            break;
        case 0b10000010:
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            break;
        case 0b00101000:
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            break;
        case 0b01001000:
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b10000100:
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            break;
        case 0b00010100:
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            break;
        case 0b01000001:
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b10000001:
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            break;
        case 0b01000010:
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b00100100:
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            break;
        case 0b00011000:
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            break;

        case 0b11110110: 
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            INV; break;
        case 0b11111001:
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            INV; break;
        case 0b01101111:
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            INV; break;
        case 0b10011111:
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            INV; break;
        case 0b11011110:
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            break;
        case 0b11101101:
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            INV; break;
        case 0b01111101:
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            INV; break;
        case 0b11010111:
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            INV; break;
        case 0b10110111:
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            INV; break;
        case 0b01111011:
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            INV; break;
        case 0b11101011:
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            INV; break;
        case 0b10111110:
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b01111110:
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            INV; break;
        case 0b10111101:
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            INV; break;
        case 0b11011011:
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            INV; break;
        case 0b11100111:
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            INV; break;

        case 0b01000011: 
            CALL(case3, 0, 1, 2, 3, 4, 5, 6, 7); 
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b10000011: 
            CALL(case3, 0, 1, 2, 3, 4, 5, 6, 7); 
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            break;
        case 0b00011010:
            CALL(case3, 1, 3, 0, 2, 5, 7, 4, 6); 
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            break;
        case 0b01001010: 
            CALL(case3, 1, 3, 0, 2, 5, 7, 4, 6); 
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b00011100: 
            CALL(case3, 3, 2, 1, 0, 7, 6, 5, 4); 
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            break;
        case 0b00101100: 
            CALL(case3, 3, 2, 1, 0, 7, 6, 5, 4); 
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            break;
        case 0b00100101:
            CALL(case3, 2, 0, 3, 1, 6, 4, 7, 5); 
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            break;
        case 0b10000101:
            CALL(case3, 2, 0, 3, 1, 6, 4, 7, 5); 
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            break;
        case 0b00110100: 
            CALL(case3, 4, 5, 0, 1, 6, 7, 2, 3);
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            break;
        case 0b00111000:
            CALL(case3, 4, 5, 0, 1, 6, 7, 2, 3); 
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            break;
        case 0b10100001: 
            CALL(case3, 5, 7, 1, 3, 4, 6, 0, 2);
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            break;
        case 0b10100100: 
            CALL(case3, 5, 7, 1, 3, 4, 6, 0, 2);
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            break;
        case 0b11000001: 
            CALL(case3, 7, 6, 3, 2, 5, 4, 1, 0);
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            break;
        case 0b11000010: 
            CALL(case3, 7, 6, 3, 2, 5, 4, 1, 0); 
             CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            break;
        case 0b01010010: 
            CALL(case3, 6, 4, 2, 0, 7, 5, 3, 1);
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            break;
        case 0b01011000: 
            CALL(case3, 6, 4, 2, 0, 7, 5, 3, 1); 
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            break;
        case 0b00011001: 
            CALL(case3, 4, 0, 6, 2, 5, 1, 7, 4);
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            break;
        case 0b10010001:
            CALL(case3, 4, 0, 6, 2, 5, 1, 7, 4);
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            break;
        case 0b00100110: 
            CALL(case3, 5, 1, 4, 0, 7, 3, 6, 2); 
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            break;
        case 0b01100010: 
            CALL(case3, 5, 1, 4, 0, 7, 3, 6, 2);
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b01000110: 
            CALL(case3, 6, 2, 7, 3, 4, 0, 5, 1); 
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            break;
        case 0b01100100: 
            CALL(case3, 6, 2, 7, 3, 4, 0, 5, 1);
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            break;
        case 0b10001001: 
            CALL(case3, 7, 3, 5, 1, 6, 2, 4, 0);
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            break;
        case 0b10011000: 
            CALL(case3, 7, 3, 5, 1, 6, 2, 4, 0); 
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            break;

        case 0b11000011: 
            CALL(case3, 0, 1, 2, 3, 4, 5, 6, 7);
            CALL(case3, 7, 6, 3, 2, 5, 4, 1, 0);
            break;
        case 0b01011010: 
            CALL(case3, 1, 3, 0, 2, 5, 7, 4, 6); 
            CALL(case3, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b00111100: 
            CALL(case3, 3, 2, 1, 0, 7, 6, 5, 4); 
            CALL(case3, 4, 5, 0, 1, 6, 7, 2, 3);
            break;
        case 0b10100101: 
            CALL(case3, 2, 0, 3, 1, 6, 4, 7, 5); 
            CALL(case3, 5, 7, 1, 3, 4, 6, 0, 2);
            break;
        case 0b10011001: 
            CALL(case3, 4, 0, 6, 2, 5, 1, 7, 4); 
            CALL(case3, 7, 3, 5, 1, 6, 2, 4, 0);
            break;
        case 0b01100110: 
            CALL(case3, 5, 1, 4, 0, 7, 3, 6, 2); 
            CALL(case3, 6, 2, 7, 3, 4, 0, 5, 1);
            break;

        case 0b01000111: 
            CALL(case5, 0, 1, 2, 3, 4, 5, 6, 7); 
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b10010011: 
            CALL(case5, 0, 4, 1, 5, 2, 6, 3, 7);
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            break;
        case 0b10010101: 
            CALL(case5, 0, 2, 4, 6, 1, 3, 5, 7); 
            CALL(case2, 7, 6, 3, 2, 5, 4, 1, 0);
            break;
        case 0b01001011: 
            CALL(case5, 1, 3, 0, 2, 5, 7, 4, 6); 
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b01100011:
            CALL(case5, 1, 0, 5, 4, 3, 2, 7, 6); 
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b01101010: 
            CALL(case5, 1, 5, 3, 7, 0, 4, 2, 6); 
            CALL(case2, 6, 4, 2, 0, 7, 5, 3, 1);
            break;
        case 0b00101101: 
            CALL(case5, 2, 0, 3, 1, 6, 4, 7, 5); 
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            break;
        case 0b01100101: 
            CALL(case5, 2, 6, 0, 4, 3, 7, 1, 5); 
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            break;
        case 0b01101100: 
            CALL(case5, 2, 3, 6, 7, 0, 1, 4, 5); 
            CALL(case2, 5, 1, 4, 0, 7, 3, 6, 2);
            break;
        case 0b00011110: 
            CALL(case5, 3, 2, 1, 0, 7, 6, 5, 4); 
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            break;
        case 0b10011010: 
            CALL(case5, 3, 1, 7, 5, 2, 0, 6, 4);
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            break;
        case 0b10011100: 
            CALL(case5, 3, 7, 2, 6, 1, 5, 0, 4); 
            CALL(case2, 4, 5, 0, 1, 6, 7, 2, 3);
            break;
        case 0b01111000: 
            CALL(case5, 4, 6, 5, 7, 0, 2, 1, 3); 
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            break;
        case 0b00111001: 
            CALL(case5, 4, 5, 0, 1, 6, 7, 2, 3);
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            break;
        case 0b01011001: 
            CALL(case5, 4, 0, 6, 2, 5, 1, 7, 3); 
            CALL(case2, 3, 2, 1, 0, 7, 6, 5, 4);
            break;
        case 0b10110100: 
            CALL(case5, 5, 4, 7, 6, 1, 0, 3, 2); 
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            break;
        case 0b00110110: 
            CALL(case5, 5, 1, 4, 0, 7, 3, 6, 2); 
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            break;
        case 0b10100110: 
            CALL(case5, 5, 7, 1, 3, 4, 6, 0, 2); 
            CALL(case2, 2, 0, 3, 1, 6, 4, 7, 5);
            break;
        case 0b11010010: 
            CALL(case5, 6, 7, 4, 5, 2, 3, 0, 1);
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            break;
        case 0b01010110: 
            CALL(case5, 6, 4, 2, 0, 7, 5, 3, 1);
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            break;
        case 0b11000110: 
            CALL(case5, 6, 2, 7, 3, 4, 0, 5, 1); 
            CALL(case2, 1, 3, 0, 2, 5, 7, 4, 6);
            break;
        case 0b11100001: 
            CALL(case5, 7, 5, 6, 4, 3, 1, 2, 0);
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            break;
        case 0b10101001: 
            CALL(case5, 7, 3, 5, 1, 6, 2, 4, 0);
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            break;
        case 0b11001001: 
            CALL(case5, 7, 6, 3, 2, 5, 4, 1, 0); 
            CALL(case2, 0, 1, 2, 3, 4, 5, 6, 7);
            break;

            //EVERYTHING

        case 0b11111111: CALL(case1, 0, 1, 2, 3, 4, 5, 6, 7); break;
        
        default: 
            CI_LOG_W("Unhandled case " << std::bitset<8>(cubeindex) << ", fallback to sampling");
            return volumeIntegralSampled(phi, h);
        }

        return InterpolationWeight_t{
            h*h*h*make_real4(weights[0], weights[1], weights[2], weights[3]),
            h*h*h*make_real4(weights[4], weights[5], weights[6], weights[7])
        };

#undef EDGE_INTERSECTIONS
#undef SHUFFLE
#undef CALL
    }

    Integration3D::InterpolationWeight_t Integration3D::surfaceIntegralMC(const InputPhi_t& phi, real h)
	{
		InterpolationWeight_t weights = { 0 };

		static float3 corners[8] = {
			make_float3(0, 0, 0),
			make_float3(1, 0, 0),
			make_float3(1, 1, 0),
			make_float3(0, 1, 0),
			make_float3(0, 0, 1),
			make_float3(1, 0, 1),
			make_float3(1, 1, 1),
			make_float3(0, 1, 1)
		};

		MarchingCubes::Triangle tris[5];
		MarchingCubes::Cell cell;
		cell.val[0] = phi.first.x; cell.val[1] = phi.first.y; cell.val[2] = phi.first.w; cell.val[3] = phi.first.z;
		cell.val[4] = phi.second.x; cell.val[5] = phi.second.y; cell.val[6] = phi.second.w; cell.val[7] = phi.second.z;
		int ntris = MarchingCubes::polygonize(cell, 0, tris);
		

		for (int tri = 0; tri < ntris; ++tri)
		{
			float3 vertices[3];
			for (int i=0; i<3; ++i)
				vertices[i] = (1 - tris[tri].v[i].weight) * corners[tris[tri].v[i].indexA] + tris[tri].v[i].weight * corners[tris[tri].v[i].indexB];
			real det = length(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
			real size = det * h * h;
			for (int i=0; i<3; ++i) //three-point rule
			{
				real x = vertices[i].x, y = vertices[i].y, z = vertices[i].z;
				weights.first.x  += (size/6)*(1 - x)*(1 - y)*(1 - z);
				weights.first.y  += (size/6)*x * (1 - y)*(1 - z);
				weights.first.z  += (size/6)*(1 - x)*y*(1 - z);
				weights.first.w  += (size/6)*x * y*(1 - z);
				weights.second.x += (size/6)*(1 - x)*(1 - y)*z;
				weights.second.y += (size/6)*x * (1 - y) * z;
				weights.second.z += (size/6)*(1 - x) * y * z;
				weights.second.w += (size/6)*x * y * z;
			}
		}

		return weights;
	}
}
