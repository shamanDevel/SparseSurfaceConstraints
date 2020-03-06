#define CUMAT_EIGEN_SUPPORT 1
#include "../SoftBodyGrid3D.h"

#include <cinder/app/AppBase.h>

#include "../Utils3D.h"
#include "../cuPrintf.cuh"

namespace ar3d
{
	typedef WorldGridData<real>::DeviceArray_t Grid_t;

	__global__ void AdvectLevelsetBlendingKernel(dim3 inputSize,
		const Grid_t inputSdf, const WorldGridData<real3>::DeviceArray_t displacements,
		Grid_t outputSdf, Grid_t outputWeights,
		int3 outputSize, int3 offsetDifference, real step, real kernelDenom, 
		SoftBodyGrid3D::AdvectionSettings settings)
	{
		CUMAT_KERNEL_3D_LOOP(x, y, z, inputSize)
			real value = inputSdf.coeff(x, y, z, -1);
            if (value > 1e10) continue; //way outside
			real3 v = displacements.coeff(x, y, z, -1);
			real extraWeight = value <= -settings.outerSdfThreshold || value >= settings.outerSdfThreshold
				? max(settings.outerSdfWeight, -abs(value) + settings.outerSdfThreshold)
				: 1;
			real3 p = make_real3(x, y, z) + step * v - make_real3(offsetDifference.x, offsetDifference.y, offsetDifference.z);
			//cuPrintf("cell=(%d,%d,%d): sdf=%5.3f, v=(%5.3f, %5.3f, %5.3f) -> p=(%5.3f, %5.3f, %5.3f)\n", x, y, z, value, v.x, v.y, v.z, p.x, p.y, p.z);
			for (int iz = max(0, (int)floor(p.z - settings.kernelRadiusOut)); iz <= min(outputSize.z - 1, (int)ceil(p.z + settings.kernelRadiusOut)); ++iz)
				for (int iy = max(0, (int)floor(p.y - settings.kernelRadiusOut)); iy <= min(outputSize.y - 1, (int)ceil(p.y + settings.kernelRadiusOut)); ++iy)
					for (int ix = max(0, (int)floor(p.x - settings.kernelRadiusOut)); ix <= min(outputSize.x - 1, (int)ceil(p.x + settings.kernelRadiusOut)); ++ix)
					{
						real d = lengthSquared3(make_real3(ix, iy, iz) - p);
						cuMat::Index idx = outputSdf.index(ix, iy, iz);
						if (d <= ar3d::utils::square(settings.kernelRadiusIn))
						{
							real w = exp(-d * kernelDenom) * extraWeight;
							atomicAddReal(outputSdf.data() + idx, w * value);
							atomicAddReal(outputWeights.data() + idx, w);
						}
						else if (d <= ar3d::utils::square(settings.kernelRadiusOut))
						{
							real w = exp(-d * kernelDenom) * settings.outerKernelWeight * extraWeight;
							atomicAddReal(outputSdf.data() + idx, w * value);
							atomicAddReal(outputWeights.data() + idx, w);
						}
					}
		CUMAT_KERNEL_3D_LOOP_END
	}

	struct AdvectionNormalizeFunctor
	{
	private:
		real outsideValue_;
	public:
		AdvectionNormalizeFunctor(real outsideValue) : outsideValue_(outsideValue) {}
        typedef real ReturnType;
		__device__ CUMAT_STRONG_INLINE real operator()(const real& output, const real& weight, cuMat::Index row, cuMat::Index col, cuMat::Index batch) const
		{
			return weight == 0 ? outsideValue_ : output / weight;
		}
	};

	void SoftBodyGrid3D::advectLevelset(const Input& input, const WorldGridData<real3>::DeviceArray_t& gridDisp,
		WorldGridDataPtr<real> advectSdf, const AdvectionSettings& settings)
	{
		const Eigen::Vector3i& srcSize = input.grid_->getSize();
		const Eigen::Vector3i& dstSize = advectSdf->getGrid()->getSize();
		const Eigen::Vector3i& srcOffset = input.grid_->getOffset();
		const Eigen::Vector3i& dstOffset = advectSdf->getGrid()->getOffset();
		assert(input.grid_->getVoxelResolution() == advectSdf->getGrid()->getVoxelResolution());

		Grid_t weights(dstSize.x(), dstSize.y(), dstSize.z());
		weights.setZero();
		advectSdf->getDeviceMemory().setZero();

		//settings
		int3 outputSize = make_int3(dstSize.x(), dstSize.y(), dstSize.z());
		int3 offsetDifference = make_int3(dstOffset.x() - srcOffset.x(), dstOffset.y() - srcOffset.y(), dstOffset.z() - srcOffset.z());
		real step = input.grid_->getVoxelResolution();
		real kernelDenom = 1.0 / (2 * ar3d::utils::square(settings.kernelRadiusIn / 3)); //98% of the Gaussian kernel is within kernelRadius

		//cinder::app::console() << "Reference SDF:\n" << input.referenceSdf_->getDeviceMemory() << std::endl;

		//blend into output grid
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(srcSize.x(), srcSize.y(), srcSize.z(), AdvectLevelsetBlendingKernel);
		AdvectLevelsetBlendingKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(
			cfg.virtual_size, input.referenceSdf_->getDeviceMemory(), gridDisp,
			advectSdf->getDeviceMemory(), weights,
			outputSize, offsetDifference, step, kernelDenom, settings);
		CUMAT_CHECK_ERROR();
		//cudaPrintfDisplay(cinder::app::console());
		CI_LOG_D("Advection: Values blended into output");

		//cinder::app::console() << "Advected SDF:\n" << advectSdf->getDeviceMemory() << std::endl;

		//normalize
		real outsideValue = dstSize.x() + dstSize.y() + dstSize.z();
		advectSdf->getDeviceMemory().inplace() = cuMat::BinaryOp<Grid_t, Grid_t, AdvectionNormalizeFunctor>
			(advectSdf->getDeviceMemory(), weights, AdvectionNormalizeFunctor(outsideValue));
		CI_LOG_D("Advection: SDF normalized");

		//cinder::app::console() << "Final SDF:\n" << advectSdf->getDeviceMemory() << std::endl;
	}
}
