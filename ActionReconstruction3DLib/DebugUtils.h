#pragma once

#include "Commons3D.h"
#include "helper_matrixmath.h"

#include <Eigen/Core>

namespace ar3d {

	class DebugUtils
	{
	public:
        //copies the blocked sparse matrix to eigen
		static Eigen::MatrixXf matrixToEigen(const SMatrix3x3& mat);
        //copies the blocked vector to eigen
		static Eigen::VectorXf vectorToEigen(const Vector3X& vec);

        //copies the eigen vector into the blocked cuMat vector
		static void eigenToVector(const Eigen::VectorXf& src, Vector3X& target);
        static Vector3X eigenToVector(const Eigen::VectorXf& src);
        //copies the eigen matrix into the blocked cuMat matrix with predefined sparsity
        static void eigenToMatrix(const Eigen::MatrixXf& src, SMatrix3x3& target);
        //Creates a dense sparse matrix and copies the eigen matrix into it
        static  SMatrix3x3 eigenToMatrix(const Eigen::MatrixXf& src);

        static void saveToMatlab(const SMatrix3x3& mat, const std::string& filename);
        static void saveToMatlab(const Vector3X& vec, const std::string& filename);

        //Makes the matrix symmetric by averaging the upper and lower triangular part: return 0.5*(A+A')
        static SMatrix3x3 makeSymmetric(const SMatrix3x3& A);

	private:
		DebugUtils() = delete;
		~DebugUtils() = default;
	};

}
