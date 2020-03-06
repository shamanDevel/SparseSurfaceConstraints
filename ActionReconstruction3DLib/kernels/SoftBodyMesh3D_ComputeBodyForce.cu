#include "../SoftBodyMesh3D.h"

namespace ar3d
{
	__global__ void ComputeBodyForceKernel(dim3 size, const int4* indices, const real3* positions, int numFree, real3 gravity, real3* forces)
	{
		CUMAT_KERNEL_1D_LOOP(i, size)
			int4 idx = indices[i];
			real3 a = positions[idx.x];
			real3 b = positions[idx.y];
			real3 c = positions[idx.z];
			real3 d = positions[idx.w];
			real area = SoftBodyMesh3D::tetSize(a, b, c, d);
			assert(area > 0);

			//TODO: neumann forces

			//gravity
			real3 f = gravity * area;
			//printf("element %lld -> force %f,%f,%f\n", i, f.x, f.y, f.z);

			//write into array
			if (idx.x < numFree /* && a.x>0 && a.y<0.7 */) atomicAddReal3(forces + idx.x, f);
			if (idx.y < numFree /* && b.x>0 && b.y<0.7 */) atomicAddReal3(forces + idx.y, f);
			if (idx.z < numFree /* && c.x>0 && c.y<0.7 */) atomicAddReal3(forces + idx.z, f);
			if (idx.w < numFree /* && d.x>0 && d.y<0.7 */) atomicAddReal3(forces + idx.w, f);
		CUMAT_KERNEL_1D_LOOP_END
	}

	void SoftBodyMesh3D::computeBodyForces(const Input& input, const Settings& settings, Vector3X& bodyForces)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(input.numElements_, ComputeBodyForceKernel);
		const int4* indices = input.indices_.data();
		const real3* positions = input.referencePositions_.data();
		real3* forces = bodyForces.data();
#if 1
		ComputeBodyForceKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(
			cfg.virtual_size, indices, positions, input.numFreeNodes_, settings.gravity_, forces);
#else
		bodyForces.inplace() = Vector3X::Constant(bodyForces.rows(), settings.gravity_);
#endif
		CUMAT_CHECK_ERROR();
	}
}