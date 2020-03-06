#include "../SoftBodyGrid3D.h"

namespace ar3d
{
	__global__ void GridComptueInitialVelocityKernel(dim3 size,
		const Vector3X referencePositions, Vector3X velocities,
		const real3 linearVelocity, const real3 angularVelocity, const real3 centerOfMass)
	{
		CUMAT_KERNEL_1D_LOOP(i, size)
			real3 pos = referencePositions.getRawCoeff(i);
			real3 vel = SoftBodySimulation3D::computeInitialVelocity(pos, centerOfMass, linearVelocity, angularVelocity);
			velocities.setRawCoeff(i, vel);
		CUMAT_KERNEL_1D_LOOP_END
	}

	void SoftBodyGrid3D::computeInitialVelocity(const Input & input, const Settings & settings, Vector3X & velocities)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(input.numActiveNodes_, GridComptueInitialVelocityKernel);
		GridComptueInitialVelocityKernel
			<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
			(cfg.virtual_size, input.referencePositions_, velocities,
				settings.initialLinearVelocity_, settings.initialAngularVelocity_, input.centerOfMass_);
		CUMAT_CHECK_ERROR();
	}
}