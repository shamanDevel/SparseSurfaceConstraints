#pragma once

//Common typedefs for 3D computations

#include "helper_math.h"
#include <cuMat/Core>
#include <cuMat/Sparse>

#ifndef AR3D_USE_DOUBLE_PRECISION
// 0 -> float, 1 -> double
#define AR3D_USE_DOUBLE_PRECISION 1
#endif

namespace ar3d
{
#if AR3D_USE_DOUBLE_PRECISION==0
    typedef float real;
    typedef float2 real2; //cuda build-ins
    typedef float4 real3; //float4 for alignment
    typedef float4 real4;
    CUMAT_STRONG_INLINE __host__ __device__ real2 make_real2(real x, real y) {return make_float2(x, y);}
    CUMAT_STRONG_INLINE __host__ __device__ real3 make_real3(real x, real y, real z) {return make_float4(x, y, z, 0);}
	CUMAT_STRONG_INLINE __host__ __device__ real3 make_real3(real x) { return make_real3(x, x, x); }
    CUMAT_STRONG_INLINE __host__ __device__ real4 make_real4(real x, real y, real z, real w) {return make_float4(x, y, z, w);}
#else
    typedef double real;
    typedef double2 real2; //cuda build-ins
    typedef double4 real3; //float4 for alignment
    typedef double4 real4;
    CUMAT_STRONG_INLINE __host__ __device__ real2 make_real2(real x, real y) { return make_double2(x, y); }
    CUMAT_STRONG_INLINE __host__ __device__ real3 make_real3(real x, real y, real z) { return make_double4(x, y, z, 0); }
    CUMAT_STRONG_INLINE __host__ __device__ real3 make_real3(real x) { return make_real3(x, x, x); }
    CUMAT_STRONG_INLINE __host__ __device__ real4 make_real4(real x, real y, real z, real w) { return make_double4(x, y, z, w); }
#endif

	typedef cuMat::Matrix<real, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::ColumnMajor> MatrixX;
	typedef cuMat::Matrix<real, cuMat::Dynamic, 1, 1, cuMat::ColumnMajor> VectorX;
	typedef cuMat::Matrix<real,1, cuMat::Dynamic, 1, cuMat::ColumnMajor> RowVectorX;

    typedef cuMat::Matrix<real3, cuMat::Dynamic, 1, 1, cuMat::ColumnMajor> Vector3X;
    typedef cuMat::Matrix<int4, cuMat::Dynamic, 1, 1, cuMat::ColumnMajor> Vector4Xi;
    typedef cuMat::Matrix<char, cuMat::Dynamic, 1, 1, cuMat::ColumnMajor> VectorXc;

    typedef cuMat::Matrix<real, 1, 1, 1, cuMat::ColumnMajor> DeviceScalar;
	typedef cuMat::Matrix<real3, 1, 1, 1, cuMat::ColumnMajor> DeviceScalar3;

	typedef cuMat::SparseMatrix<real, 1, cuMat::CSR> SMatrix;
}

//Streaming
namespace std
{
	template < class T >
	std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
	{
		os << "[";
		for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii)
		{
			os << " " << *ii;
		}
		os << "]";
		return os;
	}
    inline std::ostream& operator << (std::ostream& os, const ar3d::real3& v)
    {
        os << "(" << v.x << "," << v.y << "," << v.z << ")";
        return os;
    }
}