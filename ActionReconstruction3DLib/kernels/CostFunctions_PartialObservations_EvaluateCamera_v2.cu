#include "../CostFunctions.h"

#include <cuda_runtime.h>
#include <cuMat/Core>
#include "Utils.h"
#include "../cuPrintf.cuh"
#include <cinder/app/AppBase.h>

#include "../TrilinearInterpolation.h"

//If set to 1, points that are not matched are assigned a constant cost of the maximal possible cost.
#define USE_DUMMY_COST_FOR_UNMATCHED_POINTS 1

//Use a smooth threshold for the point distance instead of a hard one
//If positive, it specifies the steepness in the smooth heaviside function
#define USE_SMOOTH_THRESHOLD 20

namespace ar3d
{
    typedef WorldGridData<real>::DeviceArray_t Grid_t;
    typedef WorldGridData<real3>::DeviceArray_t Disp_t;

    namespace
    {
		//Threaded over pixels
		//For speed optimizations, each warp or block should process a pixel,
		//not just each pixel
        __global__ void EvaluateCameraKernel_v2(dim3 size,
            const Grid_t referenceSdf, const real h, int3 offset, const Disp_t gridDisplacements,
            const glm::mat4 cameraMatrix, const glm::mat4 cameraInvMatrix,
            const CostFunctionPartialObservations::Image observedImage,
            Disp_t adjGridDisplacementsOut, real* costOut, int* numPointsOut,
			const real maxSdf)
        {
			CUMAT_KERNEL_2D_LOOP(x, y, size)
				//Fetch depth and compute expected position
				real depth = observedImage.coeff(x, y, 0, -1);
				if (depth <= 0) continue;
				//project to world space
				glm::vec4 screenPos(x * 2 / float(observedImage.rows()) - 1, y * 2 / float(observedImage.cols()) - 1, depth * 2 - 1, 1.0);
				glm::vec4 observedPos4 = cameraInvMatrix * screenPos;
				real3 worldPos = make_real3(observedPos4.x, observedPos4.y, observedPos4.z) / observedPos4.w;

				//search cell
				bool found = false;
				real3 finalAdjPosX[8];
				real finalSdf = 1e20;
				int3 finalPos = {};
				real finalWeight = 1;
				int3 gridSize = make_int3(referenceSdf.rows(), referenceSdf.cols(), referenceSdf.batches());
				for (int ix = 0; ix < gridSize.x - 1; ++ix) 
				for (int iy = 0; iy < gridSize.y - 1; ++iy) 
				for (int iz = 0; iz < gridSize.z - 1; ++iz)
				{
					//fetch SDF values and displacements
					real sdfX[8] = {
						referenceSdf.coeff(ix, iy, iz, -1),
						referenceSdf.coeff(ix + 1, iy, iz, -1),
						referenceSdf.coeff(ix, iy + 1, iz, -1),
						referenceSdf.coeff(ix + 1, iy + 1, iz, -1),
						referenceSdf.coeff(ix, iy, iz + 1, -1),
						referenceSdf.coeff(ix + 1, iy, iz + 1, -1),
						referenceSdf.coeff(ix, iy + 1, iz + 1, -1),
						referenceSdf.coeff(ix + 1, iy + 1, iz + 1, -1)
					};
					real3 dispX[8] = {
	                    gridDisplacements.coeff(ix, iy, iz, -1),
	                    gridDisplacements.coeff(ix+1, iy, iz, -1),
	                    gridDisplacements.coeff(ix, iy+1, iz, -1),
	                    gridDisplacements.coeff(ix+1, iy+1, iz, -1),
	                    gridDisplacements.coeff(ix, iy, iz+1, -1),
	                    gridDisplacements.coeff(ix+1, iy, iz+1, -1),
	                    gridDisplacements.coeff(ix, iy+1, iz+1, -1),
	                    gridDisplacements.coeff(ix+1, iy+1, iz+1, -1)
	                };
					const real3 posX[8] = {
						make_real3((ix + offset.x) * h, (iy + offset.y) * h, (iz + offset.z) * h) + dispX[0],
						make_real3((ix + offset.x + 1) * h, (iy + offset.y) * h, (iz + offset.z) * h) + dispX[1],
						make_real3((ix + offset.x) * h, (iy + offset.y + 1) * h, (iz + offset.z) * h) + dispX[2],
						make_real3((ix + offset.x + 1) * h, (iy + offset.y + 1) * h, (iz + offset.z) * h) + dispX[3],
						make_real3((ix + offset.x) * h, (iy + offset.y) * h, (iz + offset.z + 1) * h) + dispX[4],
						make_real3((ix + offset.x + 1) * h, (iy + offset.y) * h, (iz + offset.z + 1) * h) + dispX[5],
						make_real3((ix + offset.x) * h, (iy + offset.y + 1) * h, (iz + offset.z + 1) * h) + dispX[6],
						make_real3((ix + offset.x + 1) * h, (iy + offset.y + 1) * h, (iz + offset.z + 1) * h) + dispX[7]
					};

					real3 xyz = trilinearInverse(worldPos, posX);
					if (xyz.x < 0 || xyz.y < 0 || xyz.z < 0 || xyz.x>1 || xyz.y>1 || xyz.z>1) continue;

					real3 worldPosTest = trilinear(xyz, posX);
					if (length3(worldPos - worldPosTest) > 1e-4) {
						continue; //not converged
					}

					real sdf = trilinear(xyz, sdfX);
#if USE_SMOOTH_THRESHOLD==0
					if (abs(sdf) >= maxSdf) continue;
#endif
					
					if (isnan(sdf)) continue;
					if (found && finalSdf < abs(sdf))
						continue; //duplicate, other solution is better
					//else: pick the current solution
					found = true;
					finalSdf = abs(sdf);
					finalPos = make_int3(ix, iy, iz);

					real3 adjXYZ = make_real3(0);
					trilinearAdjoint(xyz, sdfX, sdf, adjXYZ);
					trilinearInvAdjoint(xyz, posX, adjXYZ, finalAdjPosX);
				}
				if (found) {
#if USE_SMOOTH_THRESHOLD>0
					finalWeight = 1 - 1 / (1 + std::exp(-((finalSdf / maxSdf - 1)*USE_SMOOTH_THRESHOLD)));
					//std::cout << "sdf=" << finalSdf << " -> weight=" << finalWeight << std::endl;
#endif
					atomicAddReal(costOut, finalWeight * finalSdf * finalSdf * 0.5f);
					int i = finalPos.x;
					int j = finalPos.y;
					int k = finalPos.z;
					atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i, j, k),        finalWeight * finalAdjPosX[0]);
					atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i+1, j, k),      finalWeight * finalAdjPosX[1]);
					atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i, j+1, k),      finalWeight * finalAdjPosX[2]);
					atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i+1, j+1, k),    finalWeight * finalAdjPosX[3]);
					atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i, j, k+1),      finalWeight * finalAdjPosX[4]);
					atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i+1, j, k+1),    finalWeight * finalAdjPosX[5]);
					atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i, j+1, k+1),    finalWeight * finalAdjPosX[6]);
					atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i+1, j+1, k+1),  finalWeight * finalAdjPosX[7]);
					atomicAdd(numPointsOut, 1);
				}

			CUMAT_KERNEL_2D_LOOP_END
        }
    }

    std::pair<real, int> CostFunctionPartialObservations::evaluateCameraGPU_v2(WorldGridDataPtr<real> referenceSdf,
        WorldGridData<real3>::DeviceArray_t gridDisplacements, const DataCamera& camera, const Image& observedImage,
        WorldGridData<real3>::DeviceArray_t adjGridDisplacementsOut, const real maxSdf)
    {
        DeviceScalar cost; cost.setZero();
        cuMat::Scalari numPoints; numPoints.setZero();

        const Eigen::Vector3i& size = referenceSdf->getGrid()->getSize();
        cuMat::Context& ctx = cuMat::Context::current();
        cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(
            static_cast<unsigned>(observedImage.rows()), 
			static_cast<unsigned>(observedImage.cols()),
            EvaluateCameraKernel_v2);

        EvaluateCameraKernel_v2 <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (
            cfg.virtual_size, referenceSdf->getDeviceMemory(), 
            static_cast<real>(referenceSdf->getGrid()->getVoxelSize()),
            make_int3(referenceSdf->getGrid()->getOffset().x(), referenceSdf->getGrid()->getOffset().y(), referenceSdf->getGrid()->getOffset().z()),
            gridDisplacements, camera.viewProjMatrix, camera.invViewProjMatrix, observedImage,
            adjGridDisplacementsOut, cost.data(), numPoints.data(), maxSdf);
        CUMAT_CHECK_ERROR();
        //cudaPrintfDisplay_D(cinder::app::console()); cinder::app::console() << std::flush;

        real costHost = static_cast<real>(cost);
        int numPointsHost = static_cast<int>(numPoints);

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

        return std::make_pair(costHost, numPointsHost);
    }
}

