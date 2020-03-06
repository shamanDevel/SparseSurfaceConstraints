#include "../SoftBodyGrid3D.h"

namespace ar3d
{
	__global__ void GridComputeMassKernel(dim3 size,
		Vector8X interpolationVolumeWeights, Vector4Xi mapping, real mass, real* masses)
	{
		CUMAT_KERNEL_1D_LOOP(i, size)
			real8 volumeWeights = interpolationVolumeWeights.getRawCoeff(i);

			//write into array
			int4 map = mapping.getRawCoeff(i);
			atomicAddReal(masses + map.x + 0, mass * volumeWeights.first.x);
			atomicAddReal(masses + map.x + 1, mass * volumeWeights.first.y);
			atomicAddReal(masses + map.y + 0, mass * volumeWeights.first.z);
			atomicAddReal(masses + map.y + 1, mass * volumeWeights.first.w);
			atomicAddReal(masses + map.z + 0, mass * volumeWeights.second.x);
			atomicAddReal(masses + map.z + 1, mass * volumeWeights.second.y);
			atomicAddReal(masses + map.w + 0, mass * volumeWeights.second.z);
			atomicAddReal(masses + map.w + 1, mass * volumeWeights.second.w);
		CUMAT_KERNEL_1D_LOOP_END
	}

	void SoftBodyGrid3D::computeMassMatrix(const Input& input, const Settings& settings, VectorX& lumpedMass)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(input.numActiveCells_, GridComputeMassKernel);
		real* massVector = lumpedMass.data();
#if 1
		GridComputeMassKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(
			cfg.virtual_size, input.interpolationVolumeWeights_, input.mapping_, settings.mass_, massVector);
#else
		bodyForces.inplace() = Vector3X::Constant(bodyForces.rows(), settings.gravity_);
#endif
		CUMAT_CHECK_ERROR();
	}
}