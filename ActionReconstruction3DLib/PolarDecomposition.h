#pragma once

#include "Commons3D.h"
#include "helper_matrixmath.h"

namespace ar3d
{
    /**
     * \brief Routines for the polar decomposition and it's adjoint
     */
    struct PolarDecomposition
    {
        PolarDecomposition() = delete;

        //MAIN

        /**
        * \brief Computes the 3D polar decomposition F = RS.
        * The currently best algorithm is chosen
        * \param F An arbitrary, non-singular 3x3 matrix
        * \return R: the rotational component of F (a pure rotation matrix)
        */
        static __host__ __device__ real3x3 polarDecomposition(const real3x3& F);
        
        /**
        * \brief Computes the 3D polar decomposition F = RS using an iterative algorithm.
        * 
        * Corotational Simulation of Deformable Solids, Michael Hauth & Wolfgang Strasser, eq. 4.16 and 4.17
        * 
        * \param F An arbitrary, non-singular 3x3 matrix
        * \param iterations the number of iterations to peform. 5 seems to be enough
        * \return R: the rotational component of F (a pure rotation matrix)
        */
        static __host__ __device__ real3x3 polarDecompositionIterative(const real3x3& F, int iterations = 5);
        /**
         * \brief Computes the full 3D polar decomposition F = RS.
         * It iteratively computes R and then derives S from that as well.
         * 
         * \param F An arbitrary, non-singular 3x3 matrix
         * \param R [Out] The rotational component of F (unitary, orthogonal if det(F)>0)
         * \param S [Out] The scaling component of F (positive-semidefinite symmetric)
         */
        static __host__ __device__ void polarDecompositionFullIterative(const real3x3& F, real3x3& R, real3x3& S, int iterations = 5);
        /**
        * \brief Computes the full 3D polar decomposition F = RS.
        * It uses the matrix square root to compute S and derives R from that.
        * 
        * This method is not very stable, \ref polarDecompositionFullIterative is preferable.
        *
        * \param F An arbitrary, non-singular 3x3 matrix
        * \param R [Out] The rotational component of F (unitary, orthogonal if det(F)>0)
        * \param S [Out] The scaling component of F (positive-semidefinite symmetric)
        */
        static __host__ __device__ void polarDecompositionFullSquareRoot(const real3x3& F, real3x3& R, real3x3& S);

        /**
         * \brief Computes the derivative of the polar decomposition
         *  with respect to the input matrix F.
         *  In detail, it computes dF/dR in direction X.
         * 
         * In the case of adjoint problems, the direction X is simply
         * the adjoint of R.
         * 
         * The matrices \c R and \c S are computed from \c F
         * with either \ref polarDecompositionFullIterative()
         * or \ref polarDecompositionFullSquareRoot().
         * 
         * The algorithm is based on 
         *  "Riemannian Geometry of Matrix Manifolds
         *   for Lagrangian Uncertainty Quantification
         *   of Stochastic Fluid Flows"
         * by Florian Feppon (2017).
         * \b It still contains some errors
         * 
         * \param F the input matrix F
         * \param R the rotational component of F
         * \param S the scaling component of F
         * \param X the direction in which the derivative is computed
         * \return the derivative of F with respect to R in direction X
         */
        static __host__ __device__ real3x3 adjointPolarDecompositionAnalytic(
            const real3x3& F, const real3x3& R, const real3x3& S,
            const real3x3& X);

        /**
        * \brief Computes the derivative of the polar decomposition
        *  with respect to the input matrix F.
        *  In detail, it computes dF/dR in direction X.
        *
        * In the case of adjoint problems, the direction X is simply
        * the adjoint of R.
        *
        * The matrices \c R and \c S are computed from \c F
        * with either \ref polarDecompositionFullIterative()
        * or \ref polarDecompositionFullSquareRoot().
        *
        * The algorithm is an algorithmic inversion 
        * of the iterative version to compute R.
        *
        * \param F the input matrix F
        * \param R the rotational component of F
        * \param S the scaling component of F
        * \param X the direction in which the derivative is computed
        * \return the derivative of F with respect to R in direction X
        */
        static __host__ __device__ real3x3 adjointPolarDecompositionIterative(
            const real3x3& F, const real3x3& R, const real3x3& S,
            const real3x3& X, int iterations = 5);

        //HELPER

        /**
         * \brief Computes the matrix square root of the positive-definite A.
         * It uses the Denman-Beavers iteration.
         * \param A the non-singular positive-definite matrix
         * \return the square root
         */
        static __host__ __device__ real3x3 matrixSquareRoot(const real3x3& A, int iterations=10);

        /**
         * \brief Computes the three eigenvalues of the real and symmetric matrix A.
         * The eigenvalues are sorted so that eig3<=eig2<=eig1
         * \param A the symmetric matrix
         * \param eig1 the largest eigenvalue
         * \param eig2 the middle eigenvalue
         * \param eig3 the smallest eigenvalue
         */
        static __host__ __device__ void eigenvalues(const real3x3& A, real& eig1, real& eig2, real& eig3);

        /**
         * \brief Computes the eigenvector (normalized) of A
         *  with respect to the eigenvalue \c eig1.
         * \param A the matrix
         * \param eig1 the eigenvalue of the returned eigenvector
         * \param eig2 the other eigenvalue
         * \param eig3 the other eigenvalue
         * \return the eigenvector to the eigenvalue \c eig1
         */
        static __host__ __device__ real3 eigenvector(const real3x3& A, real eig1, real eig2, real eig3);

        /**
         * \brief Computes the adjoint of \f$ B=A^{-T} $\f.
         * \param A the input matrix A
         * \param adjB the adjoint of B
         * \return the adjoint of A
         */
        static __host__ __device__ real3x3 adjointMatrixTrInv(const real3x3& A, const real3x3& adjB);
    };
}