#include "../SoftBodyGrid3D.h"

#include <cinder/app/AppBase.h>

#include "../CommonKernels.h"
#include "../Utils3D.h"
#include "../cuPrintf.cuh"

#define DEBUG_PRINT_FORCES 0

namespace ar3d
{
	__global__ void GridApplyCollisionForcesKernel(dim3 size,
		const real3* refPositions, const real3* displacements, const real3* velocities, 
		const int4* mapping, const real8* sdfs, const real8* surfaceWeights,
		real4 groundPlane, real groundStiffness, real softminAlpha, real timestep, real theta, bool stableCollision,
		real3* forces)
	{
		CUMAT_KERNEL_1D_LOOP(elementIdx, size)
			//node indices
			const int4 map = mapping[elementIdx];
			const int nodeIdx[8] = { map.x, map.x + 1, map.y, map.y + 1, map.z, map.z + 1, map.w, map.w + 1 };
			//load position + displacement + velocity + init forces
			real3 posx[8];
			real3 velx[8];
			real3 forcex[8];
			#pragma unroll
			for (int i=0; i<8; ++i)
			{
				posx[i] = refPositions[nodeIdx[i]] + displacements[nodeIdx[i]];
				velx[i] = velocities[nodeIdx[i]];
				forcex[i] = make_real3(0);
			}

			//prepare intersection points
			static int EDGES[12][2] = {
				{0, 1},
				{2, 3},
				{0, 2},
				{1, 3},
				{4, 5},
				{6, 7},
				{4, 6},
				{5, 7},
				{0, 4},
				{1, 5},
				{2, 6},
				{3, 7}
			};
			real8 phiTmp = sdfs[elementIdx];
			real phi[8] = { phiTmp.first.x, phiTmp.first.y, phiTmp.first.z, phiTmp.first.w, phiTmp.second.x, phiTmp.second.y, phiTmp.second.z, phiTmp.second.w };

			//for each of those intersections, check if they are inside the cell (valid intersections)
			//and then collide them against the ground
			bool hasForce = false;
			for (int i=0; i<12; ++i)
			{
				//find intersection
				real intersection = phi[EDGES[i][0]] / (phi[EDGES[i][0]] - phi[EDGES[i][1]]);
				if (intersection < 0 || intersection > 1 || isnan(intersection)) continue;

				//get interpolated collision point on the edge
				real3 pos = posx[EDGES[i][0]] * (1 - intersection) + posx[EDGES[i][1]] * intersection;
				real3 vel = velx[EDGES[i][0]] * (1 - intersection) + velx[EDGES[i][1]] * intersection;

				//collide them against the ground -> compute force
				real4 normalDist = SoftBodySimulation3D::groundDistance(groundPlane, pos);
				real f;
				if (stableCollision) {
					real softmin = ar3d::utils::softmin(normalDist.w, softminAlpha);
					real fCurrent = -groundStiffness * softmin;
					real distDt = SoftBodySimulation3D::groundDistanceDt(groundPlane, vel);
					real fDt = -groundStiffness * (ar3d::utils::softminDx(normalDist.w, softminAlpha) * distDt);
					real fNext = fCurrent + timestep * fDt;
					f = theta * fNext + (1 - theta) * fCurrent;
				}
				else {
					//real softmin = ar3d::utils::softmin(normalDist.w, softminAlpha);
					f = -groundStiffness * min(normalDist.w, real(0));
				}
				if (f <= 1e-10) continue;
				real3 fVec = make_real3(normalDist.x, normalDist.y, normalDist.z) * f;

				//blend them into the forces
				forcex[EDGES[i][0]] += (1 - intersection) * fVec;
				forcex[EDGES[i][1]] += intersection * fVec;
				hasForce = true;

#if DEBUG_PRINT_FORCES==1
				cuPrintf_D("element %d, edge %d -> intersection at %f with force %f\n", elementIdx, i, intersection, f);
#endif
			}

			//integrate the forces over the surface
			if (hasForce) {
				real8 sw = surfaceWeights[elementIdx];
				atomicAddReal3(forces + nodeIdx[0], forcex[0] * sw.first.x);
				atomicAddReal3(forces + nodeIdx[1], forcex[1] * sw.first.y);
				atomicAddReal3(forces + nodeIdx[2], forcex[2] * sw.first.z);
				atomicAddReal3(forces + nodeIdx[3], forcex[3] * sw.first.w);
				atomicAddReal3(forces + nodeIdx[4], forcex[4] * sw.second.x);
				atomicAddReal3(forces + nodeIdx[5], forcex[5] * sw.second.y);
				atomicAddReal3(forces + nodeIdx[6], forcex[6] * sw.second.z);
				atomicAddReal3(forces + nodeIdx[7], forcex[7] * sw.second.w);
			}
		CUMAT_KERNEL_1D_LOOP_END
	}

	void SoftBodyGrid3D::applyCollisionForces(const Input& input, const Settings& settings, const State& state,
		Vector3X& bodyForces)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(input.numActiveCells_, GridApplyCollisionForcesKernel);

		GridApplyCollisionForcesKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(
			cfg.virtual_size, 
			input.referencePositions_.data(), state.displacements_.data(), state.velocities_.data(),
			input.mapping_.data(), input.cellSdfs_.data(), input.interpolationBoundaryWeights_.data(),
			settings.groundPlane_, settings.groundStiffness_, settings.softmaxAlpha_, settings.timestep_, settings.newmarkTheta_, settings.stableCollision_,
			bodyForces.data());
		CUMAT_CHECK_ERROR();
#if DEBUG_PRINT_FORCES==1
		cudaPrintfDisplay_D(cinder::app::console());
#endif
	}
}
