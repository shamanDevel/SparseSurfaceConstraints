#include "../AdjointSolver.h"

namespace ar3d
{
	//adjoint of c = cross(a, b)
	__device__ __inline__ void adjointCross(real3 a, real3 b, real3 adjC, real3& adjA, real3& adjB)
	{
		adjA.x += adjC.z * b.y - adjC.y * b.z;
		adjA.y += adjC.x * b.z - adjC.z * b.x;
		adjA.z += adjC.y * b.x - adjC.x * b.y;
		adjB.x += adjC.y * a.z - adjC.x * a.z;
		adjB.y += adjC.z * a.x - adjC.x * a.z;
		adjB.z += adjC.x * a.y - adjC.z * a.y;
	}

	__device__ __inline__ void adjointComputeInitialVelocity(
		const real3& position, const real3& centerOfMass, const real3& linearVelocity, const real3& angularVelocity,
		const real3& adjVelocity, real3& adjLinearVelocityOut, real3& adjAngularVelocityOut)
	{
		adjLinearVelocityOut += adjVelocity;
		real3 dummy;
		adjointCross(angularVelocity, position - centerOfMass, adjVelocity, adjAngularVelocityOut, dummy);
	}

	__global__ void AdjointComptueInitialVelocityKernel(dim3 size,
		const Vector3X referencePositions,
		const real3 linearVelocity, const real3 angularVelocity, const real3 centerOfMass,
		const Vector3X adjVelocity,
		real3* adjLinearVelocityOut, real3* adjAngularVelocityOut)
	{
		CUMAT_KERNEL_1D_LOOP(i, size)
			real3 pos = referencePositions.getRawCoeff(i);
			real3 adjVel = adjVelocity.getRawCoeff(i);
			real3 adjLinear = make_real3(0, 0, 0);
			real3 adjAngular = make_real3(0, 0, 0);
			adjointComputeInitialVelocity(pos, centerOfMass, linearVelocity, angularVelocity, adjVel, adjLinear, adjAngular);
			atomicAddReal3(adjLinearVelocityOut, adjLinear);
			atomicAddReal3(adjAngularVelocityOut, adjAngular);
		CUMAT_KERNEL_1D_LOOP_END
	}

	void AdjointSolver::adjointComputeInitialVelocity(
		const Input & input, const real3 & linearVelocity, const real3 & angularVelocity, 
		const Vector3X & adjVelocities, double3 & adjLinearVelocity, double3 & adjAngularVelocity)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D(input.numActiveNodes_, AdjointComptueInitialVelocityKernel);
		DeviceScalar3 adjLinearVelocityGpu; adjLinearVelocityGpu.setZero();
		DeviceScalar3 adjAngularVelocityGpu; adjAngularVelocityGpu.setZero();

		AdjointComptueInitialVelocityKernel
			<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
			(cfg.virtual_size, input.referencePositions_,
				linearVelocity, angularVelocity, input.centerOfMass_,
				adjVelocities,
				adjLinearVelocityGpu.data(), adjAngularVelocityGpu.data());
		CUMAT_CHECK_ERROR();

		real3 adjLinearVelocityCpu = static_cast<real3>(adjLinearVelocityGpu);
		real3 adjAngularVelocityCpu = static_cast<real3>(adjAngularVelocityGpu);
		adjLinearVelocity += make_double3(adjLinearVelocityCpu.x, adjLinearVelocityCpu.y, adjLinearVelocityCpu.z);
		adjAngularVelocity += make_double3(adjAngularVelocityCpu.x, adjAngularVelocityCpu.y, adjAngularVelocityCpu.z);
	}
}