#pragma once

#include <Eigen/Dense>

#include "Commons.h"

namespace ar {

    /**
     * \brief Adjoint methods of utility functions in ar::utils, SoftBodySimulation, ...
     */
    struct AdjointUtilities
    {
        /**
        * \brief Computes the adjoint of the polar decomposition
        *  \ref SoftBodySimulation::polarDecomposition(const Matrix2&).
        * \param F input matrix F
        * \param adjR adjoint of the output matrix
        * \return adjoint of the input matrix
        */
        static Matrix2 polarDecompositionAdjoint(const Matrix2& F, const Matrix2& adjR);

        /**
        * \brief Adjoint code of \ref SoftBodyMesh2D::resolveSingleCollision_PostRepair.<br>
        * Forward input: dist, normal, u, uDot, alpha, beta.<br>
        * Forward output: uNew, uDotNew.<br>
        * Adjoint output (aka in): adjUNew, adjUDotNew.<br>
        * Adjoint input (aka out): adjDist, adjNormal, adjU, adjUDot, adjAlpha, adjBeta.
        */
        [[deprecated("Implicit Spring Forces should be used instead")]]
        static void resolveSingleCollision_PostRepair_Adjoint(
            real dist, const Vector2& normal, const Vector2& u, const Vector2& uDot, real alpha, real beta,
            const Vector2& uNew, const Vector2& uDotNew, const Vector2& adjUNew, const Vector2& adjUDotNew,
            real& adjDist, Vector2& adjNormal, Vector2& adjU, Vector2& adjUDot, real& adjAlpha, real& adjBeta);

        /**
        * \brief Adjoint code of \ref SoftBodySimulation::groundCollision.<br>
        * Forward input: p, groundHeight, groundAngle.<br>
        * Adjoint output (aka in): adjDist, adjNormal.<br>
        * Adjoint input (aka out): adjP, adjGroundHeight, adjGroundAngle
        */
        static void groundCollisionAdjoint(
            const Vector2& p, real groundHeight, real groundAngle,
            real adjDist, const Vector2& adjNormal,
            Vector2& adjP, real& adjGroundHeight, real& adjGroundAngle);

        /**
        * \brief Adjoint code of \ref SoftBodySimulation::groundCollisionDt.<br>
        * Forward input: pDot (velocity), groundHeight, groundAngle.<br>
        * Adjoint output (aka in): adj d/dt dist .<br>
        * Adjoint input (aka out): adjPDot, adjGroundHeight, adjGroundAngle
        */
        static void groundCollisionDtAdjoint(const Vector2& pDot, real groundHeight, real groundAngle,
            real adjDistDt,
            Vector2& adjPDot, real& adjGroundHeight, real& adjGroundAngle);

        /**
		 * \brief Adjoint code of \code ar::utils::bilinearInterpolate(values, fx, fy) \endcode
		 *  with respect to the corner values
		 * \tparam T 
		 * \tparam S 
		 * \param fx 
		 * \param fy 
		 * \param adjResult 
		 * \param adjValues 
		 */
		template <typename T, typename S>
		static void bilinearInterpolateAdjoint1(S fx, S fy, const T& adjResult, std::array<T, 4>& adjValues)
		{
			adjValues[0] += (1 - fx) * (1 - fy) * adjResult;
			adjValues[1] += fx * (1 - fy) * adjResult;
			adjValues[2] += (1 - fx) * fy * adjResult;
			adjValues[3] += fx * fy * adjResult;
		}

        /**
         * \brief Adjoint code of \code ar::utils::bilinearInterpolate(values, fx, fy) \endcode
         *  with respect to the interpolation weights fx, fy.
         * \tparam T 
         * \tparam S 
         * \param values 
         * \param fx 
         * \param fy 
         * \param adjResult 
         * \param adjFx 
         * \param adjFy 
         */
        static void bilinearInterpolateAdjoint2(const std::array<ar::real, 4>& values, real fx, real fy,
                                                const real& adjResult, real& adjFx, real& adjFy);


        /**
         * \brief Computes the adjoint of \code mInv = m.inverse() \endcode.
         * \param m the matrix
         * \param adjMInv the adjoint of the inverse matrix
         * \return the adjoint of the matrix
         */
        static Matrix2 inverseAdjoint(const Matrix2& m, const Matrix2& adjMInv);

        /**
         * \brief Adjoint code of \code alphaBeta = ar::utils::bilinearInterpolateInv(values, x) \endcode
         *  with respect to the corner values.
         * \param values the forward corner values
         * \param x the interpolation point
         * \param alphaBeta the interpolation weights, the result of the forward step
         * \param adjAlphaBeta the adjoint of the interpolation weights (weights were computed by the forward problem)
         * \param adjValues output: the adjoint of the corner values
         */
        static void bilinearInterpolateInvAdjoint(const std::array<Vector2, 4>& values, const Vector2& x, const Vector2& alphaBeta,
			const Vector2& adjAlphaBeta, std::array<Vector2, 4>& adjValues);

        /**
         * \brief Adjoint code of \code ar::utils::softmin<real>(x, alpha) \endcode.<br>
         * Forward input: x, alpha <br>
         * Adjoint output: adjResult <br>
         * Adjoint input: adjX, <br>
         * Alpha is a meta-parameter, no adjoint computed
         */
        static void softminAdjoint(real x, real alpha, real adjResult, real& adjX);

        /**
        * \brief Adjoint code of \code ar::utils::softminDx<real>(x, alpha) \endcode.<br>
        * Forward input: x, alpha <br>
        * Forward output: result <br>
        * Adjoint output: adjResult <br>
        * Adjoint input: adjX, <br>
        * Alpha is a meta-parameter, no adjoint computed
        */
        static void softminDxAdjoint(real x, real alpha, real adjResult, real& adjX);

		/**
		* \brief Computes the derivate of the material parameters (Lamé coefficients)
		* with respect to the youngs modulus.
		* \param youngsModulus
		* \param poissonRatio
		* \param muDyoung
		* \param lambdaDyoung
		*/
		static void computeMaterialParameters_D_YoungsModulus(real youngsModulus,
			real poissonRatio, real& muDyoung, real& lambdaDyoung);

		/**
		* \brief Computes the derivate of the material parameters (Lamé coefficients)
		* with respect to the Poisson ratio.
		* \param youngsModulus
		* \param poissonRatio
		* \param muDpoisson
		* \param lambdaDpoisson
		*/
		static void computeMaterialParameters_D_PoissonRatio(real youngsModulus,
			real poissonRatio, real& muDpoisson, real& lambdaDpoisson);
    
	    /**
		 * \brief Adjoint code of Integration::getIntersectionPoints .
		 * \param sdf input SDF
		 * \param corners input corner positions
		 * \param adjPoint1 adjoint of the first intersection point
		 * \param adjPoint2 adjoint of the second intersection point
		 * \param adjSdf output: adjoint of the SDF
		 * \param adjCorners output: adjoint of the corner positions
		 */
		static void getIntersectionPointsAdjoint(
			const std::array<real, 4>& sdf, const std::array<Vector2, 4>& corners,
			const Vector2& adjPoint1, const Vector2& adjPoint2,
			std::array<real, 4>& adjSdf, std::array<Vector2, 4>& adjCorners);


    };

}