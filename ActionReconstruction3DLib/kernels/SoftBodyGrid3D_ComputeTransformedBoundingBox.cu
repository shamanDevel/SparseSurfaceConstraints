#include "../SoftBodyGrid3D.h"

#include "../Commons3D.h"
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include "../cuPrintf.cuh"
#include <cinder/app/AppBase.h>

namespace ar3d
{
	typedef real8 MinMaxType;

	struct CTBB_Functor : public thrust::unary_function<int, MinMaxType>
	{
	private:
		const real3* positions_;
		const real3* displacements_;

	public:
		__host__ CTBB_Functor(const real3* const positions, const real3* const displacements)
			: positions_(positions), displacements_(displacements)
		{}
		__device__ MinMaxType operator()(const int& idx) const
		{
			MinMaxType ret;
			real3 pos = positions_[idx] + displacements_[idx];
			ret.first = pos;
			ret.second = pos;
			//cuPrintf_D("idx=%d -> pos %5.3f %5.3f %5.3f\n", idx, pos.x, pos.y, pos.z);
			return ret;
		}
	};

	struct CTBB_Comparator : public thrust::binary_function<MinMaxType, MinMaxType, MinMaxType>
	{
		__device__ MinMaxType operator()(const MinMaxType& l, const MinMaxType& r) const
		{
			MinMaxType ret;
#if AR3D_USE_DOUBLE_PRECISION==0
			ret.first = fminf(l.first, r.first);
			ret.second = fmaxf(l.second, r.second);
#else
            ret.first = fmin(l.first, r.first);
            ret.second = fmax(l.second, r.second);
#endif
			return ret;
		}
	};

	ar::geom3d::AABBBox SoftBodyGrid3D::computeTransformedBoundingBox(const Input& input, const State& state)
	{
		MinMaxType init;
		init.first = make_real3(1e20); //min
		init.second = make_real3(-1e20); //max
		MinMaxType ret = thrust::transform_reduce(
			thrust::device,
			thrust::counting_iterator<int>(0),
			thrust::counting_iterator<int>(input.numActiveNodes_),
			CTBB_Functor(input.referencePositions_.data(), state.displacements_.data()),
			init,
			CTBB_Comparator()
		);
		//cudaPrintfDisplay_D(cinder::app::console() << "computeTransformedBoundingBox:\n");
		CI_LOG_D("Bounding box: [" << ret.first.x << " " << ret.first.y << " " << ret.first.z << "] - [" << ret.second.x << " " << ret.second.y << " " << ret.second.z << "]");
		return ar::geom3d::AABBBox(Eigen::Vector3d(ret.first.x, ret.first.y, ret.first.z), Eigen::Vector3d(ret.second.x, ret.second.y, ret.second.z));
	}

    ar::geom3d::AABBBox SoftBodyGrid3D::limitBoundingBox(const ar::geom3d::AABBBox& newBox,
        const ar::geom3d::AABBBox& oldBox)
    {
        double oldSize = (oldBox.max - oldBox.min).prod();
        double newSize = (newBox.max - newBox.min).prod();
        if (newSize > oldSize * 100)
        {
            CI_LOG_E("Divergence, the bounding box grew to large. Old volume: " << oldSize << ", new volume: " << newSize);
            return oldBox; //simply return the old box
        }
        return newBox; //no change
    }
}
