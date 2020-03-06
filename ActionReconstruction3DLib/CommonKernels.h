#pragma once

#include "Commons3D.h"
#include "helper_matrixmath.h"

namespace ar3d
{
	struct CommonKernels
	{
		
		/**
		 * \brief Solves the linear system Ax=b
		 * \param A [In] the matrix A
		 * \param b [In] the vector b
		 * \param x [In-Out] the target vector x and initial guess for x
		 * \param iterations [In] the maximal number of iterations - [Out] the actual number of iterations
		 * \param tolerance [In] the tolerance - [Out] the final error
		 */
		static void solveCG(const SMatrix3x3& A, const Vector3X& b, Vector3X& x,
			int& iterations, real& tolerance);

		/**
		 * \brief Computes the adjoint of the linear solver Ax=b.
		 * The matrix A is assumed to be symmetric
		 * \param A [In] the matrix A
		 * \param b [In] the vector b
		 * \param x [In] the solution of the problem Ax=b
		 * \param adjX [In] the adjoint of the solution x
		 * \param adjA [Out] the adjoint of the matrix A (initialized with the correct sparsity pattern and zeroed data)
		 * \param adjB [Out] the adjoint of the vector b (zeroed)
		 * \param iterations [In] the number of iterations
		 * \param tolerance [In] the tolerance
		 * \return true if the solver converged
		 */
		static bool adjointSolveCG(
			const SMatrix3x3& A, const Vector3X& b, const Vector3X& x,
			const Vector3X& adjX, SMatrix3x3& adjA, Vector3X& adjB,
			int iterations, real tolerance);

		static constexpr real NewmarkTheta = real(0.6);

		/**
		 * \brief Performs the Newmark Time Integration.
		 * 
		 * 1. Call newmarkTimeIntegration to compute A and b
		 * 2. Solve for x, the current displacements, with solveCG
		 * 3. Call newmarkComputeVelocity to compute the new velocities
		 * 
		 * \param stiffness [In] Stiffness matrix
		 * \param forces [In] force vector
		 * \param mass [In] lumped mass matrix (diagonal vector)
		 * \param prevDisplacement [In] displacements at the last timestep
		 * \param prevVelocity [In] velocities at the last timestep
		 * \param dampingMass [In] Rayleigh damping on the mass
		 * \param dampingStiffness [In] Rayleigh damping on the stiffness
		 * \param timestep [In] The timestep length
		 * \param A [Out] matrix for the linear solver (already set up with the sparsity pattern)
		 * \param b [Out] right-hand-side for the linear solver (already set up with the correct size)
		 * \param theta [In] Newmark theta, default is \ref NewmarkTheta
		 * \see newmarkComputeVelocity
		 */
		static void newmarkTimeIntegration(
			const SMatrix3x3& stiffness, const Vector3X& forces, const VectorX& mass,
			const Vector3X& prevDisplacement, const Vector3X& prevVelocity,
			real dampingMass, real dampingStiffness, real timestep,
			SMatrix3x3& A, Vector3X& b,
			real theta);

	    /**
         * \brief Adjoint of the newmark time integration matrix assembly.
         * The stiffness matrix is assumed to be symmetric.
         * \param stiffness [In]
         * \param forces [In]
         * \param mass [In]
         * \param prevDisplacement [In]
         * \param prevVelocity [In]
         * \param dampingMass [In]
         * \param dampingStiffness [In] 
         * \param adjA [In]
         * \param adjB [In]
         * \param adjStiffness [Out]
         * \param adjForces [Out]
         * \param adjMass [Out]
         * \param adjPrevDisplacement [Out]
         * \param adjPrevVelocity [Out]
         * \param adjDampingMass [Out]
         * \param adjDampingStiffness [Out]
         * \param timestep 
         * \param theta 
         */
        static void adjointNewmarkTimeIntegration(
            const SMatrix3x3& stiffness, const Vector3X& forces, const VectorX& mass,
            const Vector3X& prevDisplacement, const Vector3X& prevVelocity,
            real dampingMass, real dampingStiffness,
            const SMatrix3x3& adjA, const Vector3X& adjB,
            SMatrix3x3& adjStiffness, Vector3X& adjForces, VectorX& adjMass,
            Vector3X& adjPrevDisplacement, Vector3X& adjPrevVelocity,
            DeviceScalar& adjDampingMass, DeviceScalar& adjDampingStiffness,
            real timestep, real theta);

		/**
		 * \brief Computes the velocities after solving the linear system.
		 * 
		 * 1. Call newmarkTimeIntegration to compute A and b
		 * 2. Solve for x, the current displacements, with solveCG
		 * 3. Call newmarkComputeVelocity to compute the new velocities
		 * 
		 * \param prevDisplacement [In] displacements from the previous timestep
		 * \param prevVelocity [In] velocities from the previous timestep
		 * \param currentDisplacement [In] current displacmenets, the solution from the CG solve
		 * \param currentVelocity [Out] current velocities
		 * \param timestep [In] time step size
		 * \param theta [In] Newmark theta, default is \ref NewmarkTheta
		 * \see newmarkTimeIntegration
		 */
		static void newmarkComputeVelocity(
			const Vector3X& prevDisplacement, const Vector3X& prevVelocity,
			const Vector3X& currentDisplacement, Vector3X& currentVelocity,
			real timestep, real theta);

		/**
		 * \brief Adjoint of the newmark velocity computation
		 *        \ref newmarkComputeVelocity
		 * \param adjCurrentVelocity [In] the adjoint of the current velocity
		 * \param adjCurrentDisplacement [Out]
		 * \param adjPrevVelocity [Out]
		 * \param adjPrevDisplacement [Out]
		 * \param timestep 
		 * \param theta 
		 */
		static void adjointNewmarkComputeVelocity(
			const Vector3X& adjCurrentVelocity,
			Vector3X& adjCurrentDisplacement, Vector3X& adjPrevVelocity, Vector3X& adjPrevDisplacement,
			real timestep, real theta);

		CommonKernels() = delete;
	};
}
