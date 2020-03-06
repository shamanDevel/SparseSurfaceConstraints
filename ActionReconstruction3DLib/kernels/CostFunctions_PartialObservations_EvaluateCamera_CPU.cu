#include "../CostFunctions.h"

#include <cmath>
#include <cuMat/Core>
#include "Utils.h"
#include "../cuPrintf.cuh"
#include <cinder/app/AppBase.h>

#include "../TrilinearInterpolation.h"

#define VERBOSE_LOGGING 0

//If set to 1, points that are not matched are assigned a constant cost of the maximal possible cost.
#define USE_DUMMY_COST_FOR_UNMATCHED_POINTS 1

//Use a smooth threshold for the point distance instead of a hard one
//If positive, it specifies the steepness in the smooth heaviside function
#define USE_SMOOTH_THRESHOLD 20

namespace ar3d
{
    typedef WorldGridData<real>::DeviceArray_t Grid_t;
    typedef WorldGridData<real3>::DeviceArray_t Disp_t;
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;

    std::pair<real, int> evaluateCameraByImage(WorldGridDataPtr<real> referenceSdf,
        const std::vector<real3>& gridDisplacements, const DataCamera& camera, const EigenMatrix& observedImage,
        std::vector<real3>& adjGridDisplacements,
		const real maxSdf)
    {
        real cost = 0;
        int numPoints = 0;
        //loop over pixels
#pragma omp parallel for schedule(dynamic,1) collapse(2)
        for (int x=0; x<observedImage.rows(); ++x)
        for (int y=0; y<observedImage.cols(); ++y)
        {
            float depth = observedImage(x, y);
            if (depth <= 0) continue;
            //project to world space
            glm::vec3 screenPos(x / float(observedImage.rows()), y / float(observedImage.cols()), depth);
            glm::vec3 worldPosGlm = camera.getWorldCoordinates(screenPos);
            real3 worldPos = make_real3(worldPosGlm.x, worldPosGlm.y, worldPosGlm.z);
#if VERBOSE_LOGGING==1
            cinder::app::console() << "Pixel (" << x << "," << y << ") -> depth=" << depth << ", worldPos=(" <<
                worldPos.x << ", " << worldPos.y << ", " << worldPos.z << "):" << std::flush;
#endif
            //search cell
            bool found = false;
            const Eigen::Vector3i& size = referenceSdf->getGrid()->getSize();
            const Eigen::Vector3i& offset = referenceSdf->getGrid()->getOffset();
            const real h = referenceSdf->getGrid()->getVoxelSize();
			real3 finalAdjPosX[8];
			real finalSdf = 1e20;
			int3 finalPos = {};
			real finalWeight = 1;
            for (int ix=0; ix<size.x()-1; ++ix) for (int iy=0; iy<size.y()-1; ++iy) for (int iz=0; iz<size.z()-1; ++iz)
            {
                real sdfX[8] = {
                    referenceSdf->atHost(ix, iy, iz),
                    referenceSdf->atHost(ix+1, iy, iz),
                    referenceSdf->atHost(ix, iy+1, iz),
                    referenceSdf->atHost(ix+1, iy+1, iz),
                    referenceSdf->atHost(ix, iy, iz+1),
                    referenceSdf->atHost(ix+1, iy, iz+1),
                    referenceSdf->atHost(ix, iy+1, iz+1),
                    referenceSdf->atHost(ix+1, iy+1, iz+1)
                };
                real3 posX[8] = {
                    make_real3(ix+offset.x(),   iy+offset.y(),   iz+offset.z()  )*h + gridDisplacements[referenceSdf->getDeviceMemory().index(ix, iy, iz)],
                    make_real3(ix+offset.x()+1, iy+offset.y(),   iz+offset.z()  )*h + gridDisplacements[referenceSdf->getDeviceMemory().index(ix+1, iy, iz)],
                    make_real3(ix+offset.x(),   iy+offset.y()+1, iz+offset.z()  )*h + gridDisplacements[referenceSdf->getDeviceMemory().index(ix, iy+1, iz)],
                    make_real3(ix+offset.x()+1, iy+offset.y()+1, iz+offset.z()  )*h + gridDisplacements[referenceSdf->getDeviceMemory().index(ix+1, iy+1, iz)],
                    make_real3(ix+offset.x(),   iy+offset.y(),   iz+offset.z()+1)*h + gridDisplacements[referenceSdf->getDeviceMemory().index(ix, iy, iz+1)],
                    make_real3(ix+offset.x()+1, iy+offset.y(),   iz+offset.z()+1)*h + gridDisplacements[referenceSdf->getDeviceMemory().index(ix+1, iy, iz+1)],
                    make_real3(ix+offset.x(),   iy+offset.y()+1, iz+offset.z()+1)*h + gridDisplacements[referenceSdf->getDeviceMemory().index(ix, iy+1, iz+1)],
                    make_real3(ix+offset.x()+1, iy+offset.y()+1, iz+offset.z()+1)*h + gridDisplacements[referenceSdf->getDeviceMemory().index(ix+1, iy+1, iz+1)]
                };
                real3 xyz = trilinearInverse(worldPos, posX);

                if (std::isnan(xyz.x))
                    cinder::app::console() << " NaN! ";

                if (xyz.x<0 || xyz.y<0 || xyz.z<0 || xyz.x>1 || xyz.y>1 || xyz.z>1) continue;
#if VERBOSE_LOGGING==1
                cinder::app::console() << " Found in cell (" << ix << "," << iy << "," << iz << "), xyz=("
                    << xyz.x << "," << xyz.y << "," << xyz.z << ") " << std::flush;
                if (found) cinder::app::console() << " DUPLICATE " << std::flush;
#endif
                real3 worldPosTest = trilinear(xyz, posX);
				if (length3(worldPos - worldPosTest) > 1e-4) {
#if VERBOSE_LOGGING==1
					cinder::app::console() << " NOT CONVERGED (" << worldPosTest.x << "," << worldPosTest.y << "," << worldPosTest.z << ") " << std::flush;
#endif
					continue;
				}

                real sdf = trilinear(xyz, sdfX);
#if USE_SMOOTH_THRESHOLD==0
				if (abs(sdf) >= maxSdf) continue;
#endif

				if (found && finalSdf < abs(sdf))
					continue; //duplicate, other solution is better
				//else: pick the current solution
				found = true;
            	finalSdf = abs(sdf);
				finalPos = make_int3(ix, iy, iz);

                real3 adjXYZ = make_real3(0);
                trilinearAdjoint(xyz, sdfX, sdf, adjXYZ);
                trilinearInvAdjoint(xyz, posX, adjXYZ, finalAdjPosX);

#if VERBOSE_LOGGING==1
                cinder::app::console() << " sdf=" << sdf << " adjXYZ=(" << adjXYZ.x << "," << adjXYZ.y << "," << adjXYZ.z << ")";
                if (ignorePoint) cinder::app::console() << " [Point ignored]";
                cinder::app::console() << std::endl;
#endif
            }
			if (found) {
#pragma omp critical
				{
#if USE_SMOOTH_THRESHOLD>0
					finalWeight = 1 - 1 / (1 + std::exp(-((finalSdf / maxSdf - 1)*USE_SMOOTH_THRESHOLD)));
					//std::cout << "sdf=" << finalSdf << " -> weight=" << finalWeight << std::endl;
#endif
					cost += finalWeight * finalSdf * finalSdf * 0.5f;
					int ix = finalPos.x;
					int iy = finalPos.y;
					int iz = finalPos.z;
					adjGridDisplacements[referenceSdf->getDeviceMemory().index(ix, iy, iz)] += finalWeight * finalAdjPosX[0];
					adjGridDisplacements[referenceSdf->getDeviceMemory().index(ix + 1, iy, iz)] += finalWeight * finalAdjPosX[1];
					adjGridDisplacements[referenceSdf->getDeviceMemory().index(ix, iy + 1, iz)] += finalWeight * finalAdjPosX[2];
					adjGridDisplacements[referenceSdf->getDeviceMemory().index(ix + 1, iy + 1, iz)] += finalWeight * finalAdjPosX[3];
					adjGridDisplacements[referenceSdf->getDeviceMemory().index(ix, iy, iz + 1)] += finalWeight * finalAdjPosX[4];
					adjGridDisplacements[referenceSdf->getDeviceMemory().index(ix + 1, iy, iz + 1)] += finalWeight * finalAdjPosX[5];
					adjGridDisplacements[referenceSdf->getDeviceMemory().index(ix, iy + 1, iz + 1)] += finalWeight * finalAdjPosX[6];
					adjGridDisplacements[referenceSdf->getDeviceMemory().index(ix + 1, iy + 1, iz + 1)] += finalWeight * finalAdjPosX[7];
					numPoints++;
				}
			}
#if VERBOSE_LOGGING==1
            else cinder::app::console() << " not found!" << std::endl;
#endif
        }

        cinder::app::console() << "=> cost=" << cost << ", numPoints=" << numPoints << std::endl;

        return std::make_pair(cost, numPoints);
    }

    std::pair<real, int> CostFunctionPartialObservations::evaluateCameraCPU(WorldGridDataPtr<real> referenceSdf,
        WorldGridData<real3>::DeviceArray_t gridDisplacements, const DataCamera& camera, const Image& observedImage,
        WorldGridData<real3>::DeviceArray_t adjGridDisplacementsOut, const real maxSdf)
    {
        //copy everything to the CPU
        referenceSdf->requireHostMemory();
        std::vector<real3> gridDisplacementsHost(gridDisplacements.size());
        gridDisplacements.copyToHost(&gridDisplacementsHost[0]);
        EigenMatrix observedImageHost = observedImage.toEigen();
        std::vector<real3> adjGridDisplacementsHost(adjGridDisplacementsOut.size());
        adjGridDisplacementsOut.copyToHost(&adjGridDisplacementsHost[0]);

#if VERBOSE_LOGGING==1
        cinder::app::console() << "---------------------------------------------------------" << std::endl;
        cinder::app::console() << "   EVALUATE CAMERA" << std::endl;
        cinder::app::console() << "---------------------------------------------------------" << std::endl;
        cinder::app::console() << "Observed image:\n" << observedImageHost << std::endl;
#endif

        auto cn = evaluateCameraByImage(
			referenceSdf, gridDisplacementsHost,
			camera, observedImageHost, adjGridDisplacementsHost,
			maxSdf);
		auto costHost = cn.first;
		auto numPointsHost = cn.second;

#if USE_DUMMY_COST_FOR_UNMATCHED_POINTS==1
		//count number of points in the input image
		int totalPointsHost = static_cast<int>((observedImage > 0).cast<int>().sum());
		real maxCost = maxSdf * maxSdf * 0.5;
		int numUnmatchedPoints = totalPointsHost - numPointsHost;
		CI_LOG_I("cost=" << costHost << ", numPoints=" << numPointsHost << ", totalPoints=" << totalPointsHost << ", maxCost=" << maxCost);
		std::cout << "cost=" << costHost << ", numPoints=" << numPointsHost << ", totalPoints=" << totalPointsHost << ", maxCost=" << maxCost << std::endl;
		costHost += std::max(int(0), numUnmatchedPoints) * maxCost;
		numPointsHost = std::max(numPointsHost, totalPointsHost);
#endif

#if VERBOSE_LOGGING==1
        cinder::app::console() << "---------------------------------------------------------" << std::endl << std::endl;
#endif

        adjGridDisplacementsOut.copyFromHost(&adjGridDisplacementsHost[0]);

		return std::make_pair(costHost, numPointsHost);
    }
}
