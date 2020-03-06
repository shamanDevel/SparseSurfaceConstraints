#include "../CostFunctions.h"

#include <cuda_runtime.h>
#include <cuMat/Core>
#include "Utils.h"
#include "../cuPrintf.cuh"
#include <cinder/app/AppBase.h>

#include "../TrilinearInterpolation.h"

//If set to 1, points that are not matched are assigned a constant cost of the maximal possible cost.
#define USE_DUMMY_COST_FOR_UNMATCHED_POINTS 1

namespace ar3d
{
    typedef WorldGridData<real>::DeviceArray_t Grid_t;
    typedef WorldGridData<real3>::DeviceArray_t Disp_t;

    namespace
    {
        __global__ void EvaluateCameraKernel(dim3 size,
            const Grid_t referenceSdf, const real h, int3 offset, const Disp_t gridDisplacements,
            const glm::mat4 cameraMatrix, const glm::mat4 cameraInvMatrix,
            const CostFunctionPartialObservations::Image observedImage,
            Disp_t adjGridDisplacementsOut, real* costOut, int* numPointsOut, 
			const real maxSdf)
        {
            CUMAT_KERNEL_3D_LOOP(i, j, k, size) //thread over cells

                // load SDF values and displacements
                real sdfX[8] = {
                    referenceSdf.coeff(i, j, k, -1),
                    referenceSdf.coeff(i+1, j, k, -1),
                    referenceSdf.coeff(i, j+1, k, -1),
                    referenceSdf.coeff(i+1, j+1, k, -1),
                    referenceSdf.coeff(i, j, k+1, -1),
                    referenceSdf.coeff(i + 1, j, k+1, -1),
                    referenceSdf.coeff(i, j + 1, k+1, -1),
                    referenceSdf.coeff(i + 1, j + 1, k+1, -1),
                };
                bool outside = false;
                for (int n = 0; n < 8 && !outside; ++n) if (sdfX[n] > 1e10) outside = true;
                if (outside) continue;
                real3 dispX[8] = {
                    gridDisplacements.coeff(i, j, k, -1),
                    gridDisplacements.coeff(i+1, j, k, -1),
                    gridDisplacements.coeff(i, j+1, k, -1),
                    gridDisplacements.coeff(i+1, j+1, k, -1),
                    gridDisplacements.coeff(i, j, k+1, -1),
                    gridDisplacements.coeff(i+1, j, k+1, -1),
                    gridDisplacements.coeff(i, j+1, k+1, -1),
                    gridDisplacements.coeff(i+1, j+1, k+1, -1)
                };

                // advected positions
//                real3 posX[8];
//#pragma unroll
//                for (int n = 0; n < 8; ++n)
//                    posX[n] = make_real3((i + offset.x) * h, (j + offset.x) * h, (k + offset.x) * h) + dispX[n];
                const real3 posX[8] = {
                    make_real3((i + offset.x) * h, (j + offset.y) * h, (k + offset.z) * h) + dispX[0],
                    make_real3((i + offset.x + 1) * h, (j + offset.y) * h, (k + offset.z) * h) + dispX[1],
                    make_real3((i + offset.x) * h, (j + offset.y + 1) * h, (k + offset.z) * h) + dispX[2],
                    make_real3((i + offset.x + 1) * h, (j + offset.y + 1) * h, (k + offset.z) * h) + dispX[3],
                    make_real3((i + offset.x) * h, (j + offset.y) * h, (k + offset.z + 1) * h) + dispX[4],
                    make_real3((i + offset.x + 1) * h, (j + offset.y) * h, (k + offset.z + 1) * h) + dispX[5],
                    make_real3((i + offset.x) * h, (j + offset.y + 1) * h, (k + offset.z + 1) * h) + dispX[6],
                    make_real3((i + offset.x + 1) * h, (j + offset.y + 1) * h, (k + offset.z + 1) * h) + dispX[7]
                };
                //cuPrintf("cell=(%3d,%3d,%3d), posX[0]=(% 6.3f,% 6.3f,% 6.3f), posX[7]=(% 6.3f,% 6.3f,% 6.3f)\n",
                //    i, j, k, posX[0].x, posX[0].y, posX[0].z, posX[7].x, posX[7].y, posX[7].z);

                //final cost and adjoint variables
                real cost = 0;
                int numPoints = 0;
                real3 adjPosX[8] = {};

                // screen positions to check
                float2 minScreen, maxScreen;
                glm::vec4 posScreen = cameraMatrix * glm::vec4(posX[0].x, posX[0].y, posX[0].z, 1.0f);
                minScreen = maxScreen = make_float2(posScreen.x, posScreen.y) / posScreen.w;
#pragma unroll
                for (int n=1; n<8; ++n)
                {
                    posScreen = cameraMatrix * glm::vec4(posX[n].x, posX[n].y, posX[n].z, 1.0f);
                    float2 posScreenDehom = make_float2(posScreen.x, posScreen.y) / posScreen.w;
                    minScreen = fminf(minScreen, posScreenDehom);
                    maxScreen = fmaxf(maxScreen, posScreenDehom);
                }
                minScreen = (minScreen + make_float2(1, 1)) / 2 * make_float2(observedImage.rows(), observedImage.cols());
                maxScreen = (maxScreen + make_float2(1, 1)) / 2 * make_float2(observedImage.rows(), observedImage.cols());

                for (int screenY = max(0, __float2int_rd(minScreen.y)); screenY <= min(int(observedImage.cols() - 1), __float2int_ru(maxScreen.y)); ++screenY)
                    for (int screenX = max(0, __float2int_rd(minScreen.x)); screenX <= min(int(observedImage.rows() - 1), __float2int_ru(maxScreen.x)); ++screenX)
                    {
                        //Fetch depth and compute expected position
                        real depth = observedImage.coeff(screenX, screenY, 0, -1);
                        if (depth <= 0) continue;
                        glm::vec4 observedPos4 = cameraInvMatrix * glm::vec4(screenX * 2 / float(observedImage.rows()) - 1, screenY * 2 / float(observedImage.cols()) - 1, depth * 2 - 1, 1.0);
                        real3 observedPos = make_real3(observedPos4.x, observedPos4.y, observedPos4.z) / observedPos4.w;

                        //Solve inverse trilinear interpolation
                        real3 xyz = trilinearInverse(observedPos, posX);
                        if (xyz.x < 0 || xyz.y < 0 || xyz.z < 0 || xyz.x>1 || xyz.y>1 || xyz.z>1) continue; //not in the current cell

                        //compute the SDF value at the current position and add to the cost
                        real sdf = trilinear(xyz, sdfX);
                        if (abs(sdf) > maxSdf) continue; //HACK: ignore if too far away
                        cost += sdf * sdf * 0.5f;
                        numPoints++;

                        //adjoint of the SDF interpolation
                        real3 adjXYZ = make_real3(0);
                        trilinearAdjoint(xyz, sdfX, sdf, adjXYZ);

                        //Pixel (19,15) -> depth=0.845289, worldPos=(-0.293733, 0.969785, 0.78828)
                        // Found in cell(7, 5, 5), xyz = (0.412533, 0.939569, 0.576559)  sdf = -0.000662278 adjXYZ = (0, 0.000121188, -0.000316075)
                        //cuPrintf_D("Pixel (%d,%d) -> depth=%.4f, worldPos=(% 5.3f,% 5.3f,% 5.3f): Found in cell (%d, %d, %d), xyz = (%.3f,%.3f,%.3f)  sdf = %.5f adjXYZ = (%.3f,%.3f,%.3f)\n",
                        //    screenX, screenY, depth, observedPos.x, observedPos.y, observedPos.z, i, j, k, xyz.x, xyz.y, xyz.z, sdf, adjXYZ.x, adjXYZ.y, adjXYZ.z);

                        //adjoint of the trilinear interpolation
                        trilinearInvAdjoint(xyz, posX, adjXYZ, adjPosX);
                    }

                bool active = sdfX[0] < 0 || sdfX[1] < 0 || sdfX[2] < 0 || sdfX[3] < 0 || sdfX[4] < 0 || sdfX[5] < 0 || sdfX[6] < 0 || sdfX[7] < 0;
                //if (active || cost>0)
                //    cuPrintf_D("cell=(%3d,%3d,%3d), minScreen=(% 6.2f,% 6.2f), maxScreen=(% 6.2f,% 6.2f), numPoints=%3d, cost=%5.3f, active=%d\n", 
                //        i, j, k, minScreen.x, minScreen.y, maxScreen.x, maxScreen.y, numPoints, cost, active?1:0);
                
                //write out adjoint values
                if (cost > 0) {
                    atomicAddReal(costOut, cost);
                    atomicAdd(numPointsOut, numPoints);
                    atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i, j, k),       adjPosX[0]);
                    atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i+1, j, k),     adjPosX[1]);
                    atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i, j+1, k),     adjPosX[2]);
                    atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i+1, j+1, k),   adjPosX[3]);
                    atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i, j, k+1),     adjPosX[4]);
                    atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i+1, j, k+1),   adjPosX[5]);
                    atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i, j+1, k+1),   adjPosX[6]);
                    atomicAddReal3(adjGridDisplacementsOut.data() + adjGridDisplacementsOut.index(i+1, j+1, k+1), adjPosX[7]);
                }

            CUMAT_KERNEL_3D_LOOP_END
        }
    }

	/*
	 * TODO: This method is a lot less precise than the CPU version.
	 * A lot of duplicate points can appear, the GPU version can't filter them out at the moment.
	 * These duplicates destroy the gradient.
	 * --> For now, the CPU version is enabled
	 */

    std::pair<real, int> CostFunctionPartialObservations::evaluateCameraGPU(WorldGridDataPtr<real> referenceSdf,
        WorldGridData<real3>::DeviceArray_t gridDisplacements, const DataCamera& camera, const Image& observedImage,
        WorldGridData<real3>::DeviceArray_t adjGridDisplacementsOut, const real maxSdf)
    {
        DeviceScalar cost; cost.setZero();
        cuMat::Scalari numPoints; numPoints.setZero();

        const Eigen::Vector3i& size = referenceSdf->getGrid()->getSize();
        cuMat::Context& ctx = cuMat::Context::current();
        cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(
            static_cast<unsigned>(size.x()-1), static_cast<unsigned>(size.y()-1), static_cast<unsigned>(size.z()-1),
            EvaluateCameraKernel);

        EvaluateCameraKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (
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
		costHost += std::max(int(0), numUnmatchedPoints) * maxCost;
		numPointsHost = std::max(numPointsHost, totalPointsHost);
#endif

        return std::make_pair(costHost, numPointsHost);
    }
}
