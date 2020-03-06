#pragma once

#include "Commons3D.h"
//#include "cuPrintf.cuh"

namespace ar3d {

struct real8
{
    real4 first, second;
};
typedef cuMat::Matrix<real8, cuMat::Dynamic, 1, 1, cuMat::ColumnMajor> Vector8X;

//-------------------------------------
// Small matrix math on build-in types
//-------------------------------------

/**
 * \brief Row-major 3x3 matrix.
 * The three rows are stored using real3 (=build-in float4).
 */
struct real3x3
{
    real3 r1, r2, r3;

    __host__ __device__ CUMAT_STRONG_INLINE real3x3() {}
	explicit __host__ __device__ CUMAT_STRONG_INLINE real3x3(real v) : r1(make_real3(v,v,v)), r2(make_real3(v,v,v)), r3(make_real3(v,v,v)) {}
    __host__ __device__ CUMAT_STRONG_INLINE real3x3(const real3& r1, const real3& r2, const real3& r3) : r1(r1), r2(r2), r3(r3) {}

	static __host__ __device__ CUMAT_STRONG_INLINE real3x3 Identity()
    {
		return real3x3(make_real3(1, 0, 0), make_real3(0, 1, 0), make_real3(0, 0, 1));
    }

	static __host__ __device__ CUMAT_STRONG_INLINE real3x3 SingleEntry(int i, int j)
    {
		real3x3 m;
		m.r1 = i == 0 ? make_real3(j == 0 ? 1 : 0, j == 1 ? 1 : 0, j == 2 ? 1 : 0) : make_real3(0, 0, 0);
		m.r2 = i == 1 ? make_real3(j == 0 ? 1 : 0, j == 1 ? 1 : 0, j == 2 ? 1 : 0) : make_real3(0, 0, 0);
		m.r3 = i == 2 ? make_real3(j == 0 ? 1 : 0, j == 1 ? 1 : 0, j == 2 ? 1 : 0) : make_real3(0, 0, 0);
		return m;
    }

	__host__ __device__ CUMAT_STRONG_INLINE real& entry(int i, int j)
    {
	    if (i==0)
	    {
			if (j == 0) return r1.x;
			if (j == 1) return r1.y;
			return r1.z;
	    } else if (i==1)
	    {
			if (j == 0) return r2.x;
			if (j == 1) return r2.y;
			return r2.z;
	    } else
	    {
			if (j == 0) return r3.x;
			if (j == 1) return r3.y;
			return r3.z;
	    }
    }
	__host__ __device__ CUMAT_STRONG_INLINE const real& entry(int i, int j) const
	{
		if (i == 0)
		{
			if (j == 0) return r1.x;
			if (j == 1) return r1.y;
			return r1.z;
		}
		else if (i == 1)
		{
			if (j == 0) return r2.x;
			if (j == 1) return r2.y;
			return r2.z;
		}
		else
		{
			if (j == 0) return r3.x;
			if (j == 1) return r3.y;
			return r3.z;
		}
	}

    __host__ __device__ CUMAT_STRONG_INLINE real3x3 operator+(const real3x3& other) const
    {
        return real3x3(r1 + other.r1, r2 + other.r2, r3 + other.r3);
    }
    __host__ __device__ CUMAT_STRONG_INLINE real3x3 operator-(const real3x3& other) const
    {
        return real3x3(r1 - other.r1, r2 - other.r2, r3 - other.r3);
    }
    __host__ __device__ CUMAT_STRONG_INLINE real3x3 operator-() const
    {
        return real3x3(-r1, -r2, -r3);
    }
	__host__ __device__ CUMAT_STRONG_INLINE real3x3& operator+=(const real3x3& other)
	{
		r1 += other.r1;
		r2 += other.r2;
		r3 += other.r3;
		return *this;
	}
    __host__ __device__ CUMAT_STRONG_INLINE real3x3& operator-=(const real3x3& other)
    {
        r1 -= other.r1;
        r2 -= other.r2;
        r3 -= other.r3;
        return *this;
    }
	__host__ __device__ CUMAT_STRONG_INLINE real3x3 operator*(const real3x3& other) const //cwise multiplication
    {
		return real3x3(r1 * other.r1, r2 * other.r2, r3 * other.r3);
    }

    __host__ __device__ CUMAT_STRONG_INLINE real det() const
    {
        return (r1.x*r2.y*r3.z + r1.y*r2.z*r3.x + r1.z*r2.x*r3.y - r1.z*r2.y*r3.x - r1.x*r2.z*r3.y - r1.y*r2.x*r3.z);
    }

    __host__ __device__ CUMAT_STRONG_INLINE real trace() const
    {
        return r1.x + r2.y + r3.z;
    }

    __host__ __device__ CUMAT_STRONG_INLINE real3x3 inverse() const
    {
        real detInv = real(1) / (r1.x*r2.y*r3.z + r1.y*r2.z*r3.x + r1.z*r2.x*r3.y - r1.z*r2.y*r3.x - r1.x*r2.z*r3.y - r1.y*r2.x*r3.z);
        return real3x3(
            detInv * make_real3(r2.y*r3.z-r2.z*r3.y, r1.z*r3.y-r1.y*r3.z, r1.y*r2.z-r1.z*r2.y),
            detInv * make_real3(r2.z*r3.x-r2.x*r3.z, r1.x*r3.z-r1.z*r3.x, r1.z*r2.x-r1.x*r2.z),
            detInv * make_real3(r2.x*r3.y-r2.y*r3.x, r1.y*r3.x-r1.x*r3.y, r1.x*r2.y-r1.y*r2.x));
    }

    __host__ __device__ CUMAT_STRONG_INLINE real3x3 transpose() const
    {
        return real3x3(
            make_real3(r1.x, r2.x, r3.x),
            make_real3(r1.y, r2.y, r3.y),
            make_real3(r1.z, r2.z, r3.z));
    }

	__host__ __device__ CUMAT_STRONG_INLINE real3x3 matmul(const real3x3& rhs) const
    {
		return real3x3(
			make_real3(dot3(r1, make_real3(rhs.r1.x, rhs.r2.x, rhs.r3.x)), dot3(r1, make_real3(rhs.r1.y, rhs.r2.y, rhs.r3.y)), dot3(r1, make_real3(rhs.r1.z, rhs.r2.z, rhs.r3.z))),
			make_real3(dot3(r2, make_real3(rhs.r1.x, rhs.r2.x, rhs.r3.x)), dot3(r2, make_real3(rhs.r1.y, rhs.r2.y, rhs.r3.y)), dot3(r2, make_real3(rhs.r1.z, rhs.r2.z, rhs.r3.z))),
			make_real3(dot3(r3, make_real3(rhs.r1.x, rhs.r2.x, rhs.r3.x)), dot3(r3, make_real3(rhs.r1.y, rhs.r2.y, rhs.r3.y)), dot3(r3, make_real3(rhs.r1.z, rhs.r2.z, rhs.r3.z)))
		);
    }

	__host__ __device__ CUMAT_STRONG_INLINE real3x3 matmulT(const real3x3& rhsTransposed) const
	{
		return real3x3(
			make_real3(dot3(r1, rhsTransposed.r1), dot3(r1, rhsTransposed.r2), dot3(r1, rhsTransposed.r3)),
			make_real3(dot3(r2, rhsTransposed.r1), dot3(r2, rhsTransposed.r2), dot3(r2, rhsTransposed.r3)),
			make_real3(dot3(r3, rhsTransposed.r1), dot3(r3, rhsTransposed.r2), dot3(r3, rhsTransposed.r3))
		);
	}

    /**
	 * \brief Multiplies vector 'right' at the right side of this matrix
	 * \param right 
	 * \return this * right
	 */
	__host__ __device__ CUMAT_STRONG_INLINE real3 matmul(const real3& right) const
    {
		return ar3d::make_real3(
			dot3(r1, right),
			dot3(r2, right),
			dot3(r3, right));
    }

	/**
	 * \brief Multiplies vector 'left' at the left side of this matrix
	 * \param left
	 * \return left^T * this
	 */
	__host__ __device__ CUMAT_STRONG_INLINE real3 matmulLeft(const real3& left) const
    {
		return ar3d::make_real3(
			dot3(left, make_real3(r1.x, r2.x, r3.x)),
			dot3(left, make_real3(r1.y, r2.y, r3.y)),
			dot3(left, make_real3(r1.z, r2.z, r3.z))
		);
    }

	/**
	 * \brief Multiplies vector 'left' at the left side of this matrix transposed
	 * \param left
	 * \return left^T * this^T
	 */
	__host__ __device__ CUMAT_STRONG_INLINE real3 matmulLeftT(const real3& left) const
	{
		return ar3d::make_real3(
			dot3(left, r1),
			dot3(left, r2),
			dot3(left, r3)
		);
	}

    /**
	 * \brief Computes vec(this).dot(vec(right)) = trace(this.transpose() * right)
	 * \param right the other matrix
	 * \return the vectorized inner product
	 */
	__host__ __device__ CUMAT_STRONG_INLINE real vecProd(const real3x3& right) const
    {
		return dot3(r1, right.r1) + dot3(r2, right.r2) + dot3(r3, right.r3);
    }

	static __host__ __device__ CUMAT_STRONG_INLINE real3x3 OuterProduct(const real3& left, const real3& right)
    {
		return real3x3(
			left.x*right,
			left.y*right,
			left.z*right
			);
    }
};
typedef cuMat::Matrix<real3x3, cuMat::Dynamic, 1, 1, cuMat::ColumnMajor> Vector3x3X;
typedef cuMat::SparseMatrix<real3x3, 1, cuMat::CSR> SMatrix3x3;

__host__ __device__ CUMAT_STRONG_INLINE real3x3 operator*(real scalar, const real3x3& mat)
{
    return real3x3(scalar*mat.r1, scalar*mat.r2, scalar*mat.r3);
}
__host__ __device__ CUMAT_STRONG_INLINE real3x3 operator*(const real3x3& mat, real scalar)
{
    return real3x3(scalar*mat.r1, scalar*mat.r2, scalar*mat.r3);
}

//Returns the skew-symmetric part
__host__ __device__ CUMAT_STRONG_INLINE real3x3 skew(const real3x3& A)
{
    return real(0.5) * (A - A.transpose());
}

struct real4x3
{
	real4 r1, r2, r3;
};
typedef cuMat::Matrix<real4x3, cuMat::Dynamic, 1, 1, cuMat::ColumnMajor> Vector4x3X;

//-------------------------------------
// Atomics for these vector + matrix types
//-------------------------------------

__device__ CUMAT_STRONG_INLINE void atomicAddReal(float* mem, const float& val)
{
    atomicAdd(mem, val);
}
__device__ CUMAT_STRONG_INLINE void atomicAddReal(double* mem, const double& val)
{
#if __CUDA_ARCH__ < 600
    unsigned long long int* address_as_ull =
        (unsigned long long int*)mem;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val +
                __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
#else
    atomicAdd(mem, val);
#endif
}

__device__ CUMAT_STRONG_INLINE void atomicAddReal3(real3* mem, const real3& val)
{
	real* m = reinterpret_cast<real*>(mem);
	atomicAddReal(m + 0, val.x);
	atomicAddReal(m + 1, val.y);
	atomicAddReal(m + 2, val.z);
}
__device__ CUMAT_STRONG_INLINE void atomicAddReal4(real4* mem, const real4& val)
{
    real* m = reinterpret_cast<real*>(mem);
    atomicAddReal(m + 0, val.x);
    atomicAddReal(m + 1, val.y);
    atomicAddReal(m + 2, val.z);
    atomicAddReal(m + 3, val.w);
}
__device__ CUMAT_STRONG_INLINE void atomicAddReal3x3(real3x3* mem, const real3x3& val)
{
	real3* m = reinterpret_cast<real3*>(mem);
	atomicAddReal3(m + 0, val.r1);
	atomicAddReal3(m + 1, val.r2);
	atomicAddReal3(m + 2, val.r3);
}

__device__ CUMAT_STRONG_INLINE real3 __shfl_down_sync_real3(unsigned mask, real3 var, unsigned int delta, int width = warpSize)
{
    return make_real3(
        __shfl_down_sync(mask, var.x, delta, width),
        __shfl_down_sync(mask, var.y, delta, width),
        __shfl_down_sync(mask, var.z, delta, width)
    );
}

__device__ CUMAT_STRONG_INLINE real3x3 __shfl_down_sync_real3x3(unsigned mask, real3x3 var, unsigned int delta, int width = warpSize)
{
    return real3x3(
        __shfl_down_sync_real3(mask, var.r1, delta, width),
        __shfl_down_sync_real3(mask, var.r2, delta, width),
        __shfl_down_sync_real3(mask, var.r3, delta, width)
    );
}

//---------------------------------
// Custom functors
//---------------------------------

//Computes v.x+v.y+v.z
struct Real3SumFunctor
{
    typedef ar3d::real ReturnType;
    __device__ CUMAT_STRONG_INLINE ar3d::real operator()(const ar3d::real3& v, cuMat::Index row, cuMat::Index col, cuMat::Index batch) const
    {
        return v.x + v.y + v.z;
    }
};

}


//---------------------------------
// CuMat functor specializations
//---------------------------------
CUMAT_NAMESPACE_BEGIN

namespace internal
{
	//num traits
	template <>
	struct NumTraits<ar3d::real3>
	{
		typedef ar3d::real3 Type;
		typedef ar3d::real3 RealType;
		typedef ar3d::real ElementalType;
		enum
		{
			IsCudaNumeric = false,
			IsComplex = false,
		};
		static constexpr CUMAT_STRONG_INLINE ar3d::real epsilon() { return std::numeric_limits<ar3d::real>::epsilon(); }
	};
	template <>
	struct NumTraits<ar3d::real3x3>
	{
		typedef ar3d::real3x3 Type;
		typedef ar3d::real3x3 RealType;
		typedef ar3d::real ElementalType;
		enum
		{
			IsCudaNumeric = false,
			IsComplex = false,
		};
		static constexpr CUMAT_STRONG_INLINE ar3d::real epsilon() { return std::numeric_limits<ar3d::real>::epsilon(); }
	};

	//real3x3 * real3 product
	template<>
	struct ProductElementFunctor<ar3d::real3x3, ar3d::real3,
		ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE>
	{
		using Scalar = ar3d::real3;
		static CUMAT_STRONG_INLINE __device__ Scalar mult(
			const ar3d::real3x3& left, const ar3d::real3& right)
		{
			return left.matmul(right);
		}
	};

	//real3 * real3 -> real3x3 outer-product
	template<>
	struct ProductElementFunctor<ar3d::real3, ar3d::real3,
		ProductArgOp::NONE, ProductArgOp::TRANSPOSED, ProductArgOp::NONE>
	{
		using Scalar = ar3d::real3x3;
		static CUMAT_STRONG_INLINE __device__ Scalar mult(
			const ar3d::real3& left, const ar3d::real3& right)
		{
			return ar3d::real3x3::OuterProduct(left, right);
		}
	};

	//real3 -> diagonal real3x3
	template<>
	struct AsDiagonalFunctor<ar3d::real3>
	{
		using MatrixType = ar3d::real3x3;
		static __host__ __device__ CUMAT_STRONG_INLINE MatrixType asDiagonal(const ar3d::real3& v)
		{
			return ar3d::real3x3(
				ar3d::make_real3(v.x, 0, 0),
				ar3d::make_real3(0, v.y, 0),
				ar3d::make_real3(0, 0, v.z)
			);
		}
	};

	//diagonal real3x3 -> real3
	template<>
	struct ExtractDiagonalFunctor<ar3d::real3x3>
	{
		using VectorType = ar3d::real3;
		static __host__ __device__ CUMAT_STRONG_INLINE VectorType extractDiagonal(const ar3d::real3x3& v)
		{
			return ar3d::make_real3(v.r1.x, v.r2.y, v.r3.z);
		}
	};
}

namespace functor
{
    //real3 -> float3 cast
    template<>
    struct CastFunctor<ar3d::real3, float3>
    {
        static __device__ CUMAT_STRONG_INLINE float3 cast(const ar3d::real3& source)
        {
            return make_float3(source.x, source.y, source.z);
        }
    };

	//real -> real3 broadcasting
	template<>
	struct CastFunctor<ar3d::real, ar3d::real3>
	{
		static __device__ CUMAT_STRONG_INLINE ar3d::real3 cast(const ar3d::real& source)
		{
			return ar3d::make_real3(source, source, source);
		}
	};

	//checked inversion of real3
	template<>
	struct UnaryMathFunctor_cwiseInverseCheck<ar3d::real3>
	{
	public:
		typedef ar3d::real3 ReturnType;
		__device__ CUMAT_STRONG_INLINE ar3d::real3 operator()(const ar3d::real3& v, Index row, Index col, Index batch) const
		{
			return ar3d::make_real3(
				(v.x == ar3d::real(0) ? ar3d::real(1) : ar3d::real(1) / v.x),
				(v.y == ar3d::real(0) ? ar3d::real(1) : ar3d::real(1) / v.y),
				(v.z == ar3d::real(0) ? ar3d::real(1) : ar3d::real(1) / v.z)
			);
		}
	};

	//cwiseAbs2 (for squaredNorm), must return real
	template<>
	struct UnaryMathFunctor_cwiseAbs2<ar3d::real3>
	{
	public:
		typedef ar3d::real ReturnType;
		__device__ CUMAT_STRONG_INLINE ar3d::real operator()(const ar3d::real3& v, Index row, Index col, Index batch) const
		{
			return v.x*v.x + v.y*v.y + v.z*v.z;
		}
	};
    //cwiseAbs2 (for squaredNorm), must return real
    template<>
    struct UnaryMathFunctor_cwiseAbs2<ar3d::real3x3>
    {
    public:
        typedef ar3d::real ReturnType;
        __device__ CUMAT_STRONG_INLINE ar3d::real operator()(const ar3d::real3x3& v, Index row, Index col, Index batch) const
        {
            return dot3(v.r1, v.r1) + dot3(v.r2, v.r2) + dot3(v.r3, v.r3);
        }
    };

	//cwiseDot
	template<>
	struct BinaryMathFunctor_cwiseDot<ar3d::real3>
	{
	public:
		typedef ar3d::real ReturnType;
		__device__ CUMAT_STRONG_INLINE ar3d::real operator()(const ar3d::real3& x, const ar3d::real3& y, Index row, Index col, Index batch) const
		{
			return dot3(x, y);
		}
	};

    //cwiseDot
    template<>
    struct BinaryMathFunctor_cwiseDot<ar3d::real3x3>
    {
    public:
        typedef ar3d::real ReturnType;
        __device__ CUMAT_STRONG_INLINE ar3d::real operator()(const ar3d::real3x3& x, const ar3d::real3x3& y, Index row, Index col, Index batch) const
        {
            ar3d::real v = dot3(x.r1, y.r1) + dot3(x.r2, y.r2) + dot3(x.r3, y.r3);
            //cuPrintf("cwiseDot<real3x3>: row=%d, col=%d -> x=[%5.3f, %5.3f, %5.3f; %5.3f, %5.3f, %5.3f; %5.3f, %5.3f, %5.3f\n", int(row), int(col), float(x.r1.x), float(x.r1.y), float(x.r1.z), float(x.r2.x), float(x.r2.y), float(x.r2.z), float(x.r3.x), float(x.r3.y), float(x.r3.z));
            //cuPrintf("cwiseDot<real3x3>: row=%d, col=%d -> y=[%5.3f, %5.3f, %5.3f; %5.3f, %5.3f, %5.3f; %5.3f, %5.3f, %5.3f\n", int(row), int(col), float(y.r1.x), float(y.r1.y), float(y.r1.z), float(y.r2.x), float(y.r2.y), float(y.r2.z), float(y.r3.x), float(y.r3.y), float(y.r3.z));
            //cuPrintf("cwiseDot<real3x3>: row=%d, col=%d -> v=%5.3f\n", int(row), int(col), float(v));
            return v;
        }
    };

}

CUMAT_NAMESPACE_END

// ----------------------
// STREAMING
// ----------------------
namespace std {
    inline std::ostream& operator << (std::ostream& os, const ar3d::real3x3& v)
    {
        os << "[" << v.r1.x << "," << v.r1.y << "," << v.r1.z 
            << ";" << v.r2.x << "," << v.r2.y << "," << v.r2.z
            << ";" << v.r3.x << "," << v.r3.y << "," << v.r3.z << "]";
        return os;
    }
}