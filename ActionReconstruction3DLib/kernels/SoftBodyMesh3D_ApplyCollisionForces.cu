#include "../SoftBodyMesh3D.h"
#include "../CommonKernels.h"
#include "../Utils3D.h"

namespace ar3d
{
	__global__ void MeshApplyCollisionForcesKernel(dim3 size,
		const real3* positions, const real3* displacements, const real3* velocities,
		real4 groundPlane, real groundStiffness, real softminAlpha, real timestep, real theta,
		real3* forces)
	{
		CUMAT_KERNEL_1D_LOOP(i, size)
			real3 pos = positions[i] + displacements[i];
			real3 vel = velocities[i];
			real4 normalDist = SoftBodySimulation3D::groundDistance(groundPlane, pos);
			real softmin = ar3d::utils::softmin(normalDist.w, softminAlpha);
			real distDt = SoftBodySimulation3D::groundDistanceDt(groundPlane, vel);
			real fCurrent = -groundStiffness * softmin; //current timestep
			real fDt = -groundStiffness * (ar3d::utils::softminDx(normalDist.w, softminAlpha) * distDt); //time derivative
			real fNext = fCurrent + timestep * fDt; //next timestep
			real f = theta * fNext + (1 - theta) * fCurrent; //average force magnitude
			forces[i] += f * make_real3(normalDist.x, normalDist.y, normalDist.z); //final force
		CUMAT_KERNEL_1D_LOOP_END
	}

	void SoftBodyMesh3D::applyCollisionForces(const Input& input, const Settings& settings, const State& state,
		Vector3X& bodyForces)
	{
		//For simplicity, I assume every node has the same boundary length, which is equal to one.
		//This means, no correction for tet size is applied.
		//This has to be controlled by the groundStiffness in the settings

		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(input.numFreeNodes_, MeshApplyCollisionForcesKernel);
		
		const real3* positions = input.referencePositions_.data();
		const real3* displacements = state.displacements_.data();
		const real3* velocities = state.velocities_.data();
		real3* forces = bodyForces.data();

		MeshApplyCollisionForcesKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(
			cfg.virtual_size, positions, displacements, velocities, 
			settings.groundPlane_, settings.groundStiffness_, settings.softmaxAlpha_, settings.timestep_, settings.newmarkTheta_,
			forces);
		CUMAT_CHECK_ERROR();
	}
}
