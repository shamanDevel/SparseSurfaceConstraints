#pragma once

#include <Eigen/Core>
#include <vector>
#include "Commons.h"


namespace ar
{
	
	template<typename VectorType>
	struct ConjugateGradient_StorageForAdjoint_Step
	{
		VectorType p, q, r;
		typename VectorType::Scalar alpha, beta1, beta2, absNew, absOld;
	};
	template<typename VectorType>
	using ConjugateGradient_StorageForAdjoint_StepVector =
		std::vector<ConjugateGradient_StorageForAdjoint_Step<VectorType>,
		Eigen::aligned_allocator<ConjugateGradient_StorageForAdjoint_Step<VectorType> > >;
    /**
    * \brief Storage of some forward variables in each iteration,
    * needed for the adjoint.
    * \tparam VectorType
    */
    template<typename VectorType>
    struct ConjugateGradient_StorageForAdjoint
    {
        ConjugateGradient_StorageForAdjoint_StepVector<VectorType> steps;
        Eigen::Index iterations;
    };

	/**
	 * \brief Conjugate Gradient
	 * \tparam MatrixType The matrix that should be inverted. Must support an operator*(Vector)->Vector
	 * \tparam Rhs The right hand side vector
	 * \tparam Dest the initial and result vector
	 * \param mat the matrix
	 * \param rhs the right hand side
	 * \param x the initial and output value
	 * \param iters In: maximal number of iterations, Out: number of iterations performed
	 * \param tol_error In: maximal allowed tolerance, Out: final error
	 * \return the number of iterations that were performed
	 */
	template <typename MatrixType, typename Rhs, typename Dest>
	EIGEN_DONT_INLINE
        ConjugateGradient_StorageForAdjoint<Eigen::Matrix<typename Dest::Scalar, Eigen::Dynamic, 1> >
	ConjugateGradientForward(
		const MatrixType& mat, const Rhs& rhs, Dest& x,
		Eigen::Index& iters, typename Dest::RealScalar& tol_error)
	{
		using namespace Eigen;
		using std::sqrt;
		using std::abs;
		typedef typename Dest::RealScalar RealScalar;
		typedef typename Dest::Scalar Scalar;
		typedef Matrix<Scalar, Dynamic, 1> VectorType;
        ConjugateGradient_StorageForAdjoint<VectorType> tmpStorage;

		RealScalar tol = tol_error;
		Index maxIters = iters;

		Index n = mat.cols();
		VectorType precond = mat.diagonal().cwiseInverse();

		//initial residual
		VectorType residual = rhs - mat * x; 

		//Check if we are already converged
		RealScalar rhsNorm2 = rhs.squaredNorm();
		if (rhsNorm2 == 0)
		{
			x.setZero();
			iters = 0;
			tol_error = 0;
			return tmpStorage;
		}
		RealScalar threshold = tol * tol*rhsNorm2;
		RealScalar residualNorm2 = residual.squaredNorm();
		if (residualNorm2 < threshold)
		{
			iters = 0;
			tol_error = sqrt(residualNorm2 / rhsNorm2);
			return tmpStorage;
		}

		VectorType z(n), p(n), q(n);
		RealScalar absOld = 0;
		
		Index i = 0;
		while (i < maxIters)
		{
            ConjugateGradient_StorageForAdjoint_Step<VectorType> storage;

			//preconditioning
            storage.r = residual;
			z = precond.cwiseProduct(residual);//precond.solve(residual);

			RealScalar absNew = residual.dot(z);  // the square of the absolute value of r scaled by invM
            storage.absNew = absNew;

			//specify search direction
			Scalar beta1 = 0;
			if (i == 0)
				p = z;
			else {
				beta1 = absNew / absOld;
				p = z + beta1 * p;
			}
            storage.beta1 = beta1;
            storage.p = p;

			q.noalias() = mat * p;                 // the bottleneck of the algorithm
            storage.q = q;

			Scalar beta2 = p.dot(q);
            storage.beta2 = beta2;
			Scalar alpha = absNew / beta2;         // the amount we travel on dir
            storage.alpha = alpha;
			x += alpha * p;                        // update solution
			residual -= alpha * q;                 // update residual

            storage.absOld = absOld;
			absOld = absNew;
			i++;

            tmpStorage.steps.push_back(storage);

			residualNorm2 = residual.squaredNorm();
			if (residualNorm2 < threshold)
				break;
		}
		tol_error = sqrt(residualNorm2 / rhsNorm2);
		iters = i;

        tmpStorage.iterations = iters;
		return tmpStorage;
	}

	/**
	 * \brief Computes the adjoint of the dot product <code>result = a.dot(b)</code>
	 * \tparam VectorType the vector type
	 * \param a the first input vector
	 * \param b the second input vector
	 * \param adjResult the adjoint of the result
	 * \param adjA Out: the modified adjoint of the first input
	 * \param adjB Out: the modified adjoint of the second input
	 */
	template <typename VectorType>
	void DotProductAdjoint(const VectorType& a, const VectorType& b,
		typename VectorType::Scalar adjResult,
		VectorType& adjA, VectorType& adjB)
	{
		adjA += b * adjResult;
		adjB += a * adjResult;
	}

	/**
	 * \brief Computes the adjoint of the Conjugate Gradient Method
	 * with respect to both the matrix and the right hand side.
	 * \tparam MatrixType 
	 * \tparam AdjMatrixType 
	 * \tparam Rhs 
	 * \tparam Dest 
	 * \param matTransposed an expression of the transposed of the matrix
	 * \param x the initial guess from the forward solve
	 * \param adjRhs Inout: adjoint of the right hand side
	 * \param adjX Inout: adjoint of the result / initial value
	 * \param adjMat Inout: adjoint of the matrix
	 * \param tmpStorage storage from the forward pass
	 */
	template <typename MatrixType, typename AdjMatrixType, typename Rhs, typename Dest>
	EIGEN_DONT_INLINE
	void ConjugateGradientAdjoint(
		const MatrixType& matTransposed, const Dest& x,
		Rhs& adjRhs, Rhs& adjX, AdjMatrixType& adjMat,
		const ConjugateGradient_StorageForAdjoint<Eigen::Matrix<typename Dest::Scalar, Eigen::Dynamic, 1> >& tmpStorage)
	{
		using namespace Eigen;
		using std::sqrt;
		using std::abs;
		typedef typename Dest::RealScalar RealScalar;
		typedef typename Dest::Scalar Scalar;
		typedef Matrix<Scalar, Dynamic, 1> VectorType;

		Index n = matTransposed.rows();
		VectorType precond = matTransposed.diagonal().cwiseInverse();

		Scalar adjAbsNew = 0, adjAbsOld = 0, adjAlpha = 0, adjBeta1 = 0, adjBeta2 = 0;
		VectorType adjP = VectorType::Zero(n);
		VectorType adjZ = VectorType::Zero(n);
		VectorType adjQ = VectorType::Zero(n);
		VectorType adjR = VectorType::Zero(n);
		VectorType adjPrecond = VectorType::Zero(n);

		for (Eigen::Index k = tmpStorage.iterations-1; k >= 0; --k)
		{
			const auto& storage = tmpStorage.steps[k];

			adjAbsNew += adjAbsOld;
			adjAbsOld = 0;

			adjAlpha -= adjR.dot(storage.q);
			adjQ -= storage.alpha * adjR;

			adjAlpha += adjX.dot(storage.p);
			adjP += storage.alpha * adjX;

			adjBeta2 -= (storage.absNew / ar::square(storage.beta2)) * adjAlpha;
			adjAbsNew += adjAlpha / storage.beta2;
			adjAlpha = 0;

			DotProductAdjoint(storage.p, storage.q, adjBeta2, adjP, adjQ);
			adjBeta2 = 0;

			adjP += matTransposed * adjQ;
			adjMat += adjQ * storage.p.transpose();
			adjQ.setZero();

			if (k == 0)
			{
				adjZ += adjP;
				adjP.setZero();
			} else
			{
				adjBeta1 += adjP.dot(storage.p);
				adjZ += adjP;
				adjP = storage.beta1 * adjP;

				adjAbsOld -= (storage.absNew / ar::square(storage.absOld)) * adjBeta1;
				adjAbsNew += adjBeta1 / storage.absOld;
				adjBeta1 = 0;
			}

			VectorType z = precond.cwiseProduct(storage.r);//precond.solve(storage.r);
			DotProductAdjoint(storage.r, z, adjAbsNew, adjR, adjZ);
			adjAbsNew = 0;

			//preconditioning
			//adjR += precond.solve(adjZ);
			adjR += precond.cwiseProduct(adjZ);
			adjPrecond += storage.r.cwiseProduct(adjZ);
			adjZ.setZero();
		}

		adjRhs += adjR;
		adjX -= matTransposed * adjR;
		adjMat -= adjR * x.transpose();
		adjR.setZero();

		adjMat.diagonal() -= (precond.array() * precond.array() * adjPrecond.array()).matrix();
	}
}
