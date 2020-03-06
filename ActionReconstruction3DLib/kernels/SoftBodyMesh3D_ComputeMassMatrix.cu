#include "../SoftBodyMesh3D.h"

namespace ar3d {

	__global__ void ComputeMassMatrixKernel(dim3 size, const int4* indices, const real3* positions, int numFree, real mass, real* massVector)
	{
		CUMAT_KERNEL_1D_LOOP(i, size)
			int4 idx = indices[i];
			real3 a = positions[idx.x];
			real3 b = positions[idx.y];
			real3 c = positions[idx.z];
			real3 d = positions[idx.w];
			real area = SoftBodyMesh3D::tetSize(a, b, c, d);
			assert(area > 0);
			//printf("Element %d: mass %f\n", int(i), float(area));
			real m = mass * area / 4;
			if (idx.x < numFree) atomicAddReal(massVector + idx.x, m);
			if (idx.y < numFree) atomicAddReal(massVector + idx.y, m);
			if (idx.z < numFree) atomicAddReal(massVector + idx.z, m);
			if (idx.w < numFree) atomicAddReal(massVector + idx.w, m);
		CUMAT_KERNEL_1D_LOOP_END
	}

	void SoftBodyMesh3D::computeMassMatrix(const Input& input, const Settings& settings, VectorX& lumpedMass)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(input.numElements_, ComputeMassMatrixKernel);
		const int4* indices = input.indices_.data();
		const real3* positions = input.referencePositions_.data();
		real* massVector = lumpedMass.data();
#if 1
		ComputeMassMatrixKernel <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (
			cfg.virtual_size, indices, positions, input.numFreeNodes_, settings.mass_, massVector);
#else
		lumpedMass.inplace() = VectorX::Constant(lumpedMass.rows(), settings.mass_);
#endif
		CUMAT_CHECK_ERROR();
	}

}