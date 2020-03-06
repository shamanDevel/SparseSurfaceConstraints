#include "../SoftBodyGrid3D.h"

namespace ar3d
{
	__global__ void GridComputeBodyForceKernel(dim3 size, 
        Vector8X interpolationVolumeWeights, Vector4Xi mapping, real3 gravity, real3* forces)
	{
		CUMAT_KERNEL_1D_LOOP(i, size)
            real8 volumeWeights = interpolationVolumeWeights.getRawCoeff(i);
			real area = 
                volumeWeights.first.x + volumeWeights.first.y + volumeWeights.first.z + volumeWeights.first.w + 
                volumeWeights.second.x + volumeWeights.second.y + volumeWeights.second.z + volumeWeights.second.w;
			assert(area > 0);

			//TODO: neumann forces

			//gravity
			real3 f = gravity * area;
			//printf("element %lld -> force %f,%f,%f\n", i, f.x, f.y, f.z);

			//write into array
            int4 map = mapping.getRawCoeff(i);
            atomicAddReal3(forces + map.x, f);
            atomicAddReal3(forces + map.x + 1, f);
            atomicAddReal3(forces + map.y, f);
            atomicAddReal3(forces + map.y + 1, f);
            atomicAddReal3(forces + map.z, f);
            atomicAddReal3(forces + map.z + 1, f);
            atomicAddReal3(forces + map.w, f);
            atomicAddReal3(forces + map.w + 1, f);
		CUMAT_KERNEL_1D_LOOP_END
	}

	void SoftBodyGrid3D::computeBodyForces(const Input& input, const Settings& settings, Vector3X& bodyForces)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(input.numActiveCells_, GridComputeBodyForceKernel);
		real3* forces = bodyForces.data();
#if 1
		GridComputeBodyForceKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(
			cfg.virtual_size, input.interpolationVolumeWeights_, input.mapping_, settings.gravity_, forces);
#else
		bodyForces.inplace() = Vector3X::Constant(bodyForces.rows(), settings.gravity_);
#endif
		CUMAT_CHECK_ERROR();
	}
}